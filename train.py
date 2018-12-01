from tqdm import tqdm

import  torch
from    torch.autograd import Variable, grad
from    torchvision import utils

from    datetime import datetime
import  random
import  os

import  config
import  utils
import  data
import  evaluate
from    session import Session, accumulate
from    torch.nn import functional as F

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

args     = config.get_config()
writer   = None
bcelogit = torch.nn.BCEWithLogitsLoss()

def train(session, train_data_loader, test_data_loader, total_steps):
    pbar = tqdm(initial=session.sample_i, total = total_steps)

    encoder,generator,g_running,critic = session.encoder,session.generator,session.g_running,session.critic

    refresh_dataset = True
    batch_count     = 0

    reset_optimizers_on_phase_start = False
    epoch_len = len(train_data_loader(1,4).dataset)

    if args.step_offset != 0:
        if args.step_offset == -1:
            args.step_offset = session.sample_i
        print("Step offset is {}".format(args.step_offset))
        session.phase += args.phase_offset
        session.alpha = 0.0

    while session.sample_i < total_steps:
        #######################  Phase Maintenance ####################### 

        steps_in_previous_phases = max(session.phase * args.images_per_stage, args.step_offset)

        sample_i_current_stage = session.sample_i - steps_in_previous_phases

        # If we can move to the next phase
        if sample_i_current_stage >= args.images_per_stage:
            if session.phase < args.max_phase: # If any phases left
                iteration_levels = int(sample_i_current_stage / args.images_per_stage)
                session.phase += iteration_levels
                sample_i_current_stage -= iteration_levels * args.images_per_stage
                print("iteration B alpha={} phase {} will be reduced to 1 and [max]".format(sample_i_current_stage, session.phase))

                refresh_dataset     = True

                if reset_optimizers_on_phase_start:
                    utils.requires_grad(generator)
                    utils.requires_grad(encoder)
                    utils.requires_grad(critic)
                    generator.zero_grad()
                    encoder.zero_grad()
                    critic.zero_grad()
                    session.reset_opt()
                    print("Optimizers have been reset.")                

        reso    = session.cur_res()
        batch   = session.cur_batch()

        # If we can switch from fade-training to stable-training
        if sample_i_current_stage >= args.images_per_stage:
            refresh_dataset = True

        session.alpha = min(1, sample_i_current_stage * 4.0 / args.images_per_stage) # For 100k, it was 0.00002 = 2.0 / args.images_per_stage

        if refresh_dataset:
            train_dataset = data.Utils.sample_data2(train_data_loader, batch, reso)
            refresh_dataset = False
            print("Refreshed dataset. Alpha={} and iteration={}".format(session.alpha, sample_i_current_stage))

        stats = {}

        try:
            real_image, _ = next(train_dataset)
        except (OSError, StopIteration):
            train_dataset = data.Utils.sample_data2(train_data_loader, batch, reso)
            real_image, _ = next(train_dataset)

        if (batch_count + 1) % (args.n_critic + 1) == 0:
            utils.requires_grad(encoder, True)
            utils.requires_grad(generator, True)
            utils.requires_grad(critic, False)
        else:
            utils.requires_grad(encoder, False)
            utils.requires_grad(generator, False)
            utils.requires_grad(critic, True)

        encoder.zero_grad()
        generator.zero_grad()
        critic.zero_grad()

        x = Variable(real_image).cuda(async=(args.gpu_count > 1))
        losses = []

        real_z   = encoder(x, session.phase, session.alpha)
        fake_x   = generator(real_z, session.phase, session.alpha)
        # fake_z   = encoder(fake_x, session.phase, session.alpha)
        # fake_cls = critic(fake_z)
        fake_cls = critic(fake_x,session.phase, session.alpha)

        if (batch_count + 1) % (args.n_critic + 1) == 0:
            ###### Autoencoder update #########
            # match_x: E_x||g(e(x)) - x|| -> min_e
            err = utils.mismatch(fake_x, x, args.match_x_metric)
            # losses.append(err)
            stats['x_reconstruction_error'] = err.data

            wgan_G_loss =  -torch.log(fake_cls).mean()
            losses.append(wgan_G_loss)

            loss = sum(losses)
            stats['G_loss'] = loss.data.cpu().numpy()
            loss.backward()

            session.optimizerE.step()
            session.optimizerG.step()
            accumulate(g_running, generator)

            ########################  Statistics ########################
            xr = stats['x_reconstruction_error']
            e = (session.sample_i / float(epoch_len))
            pbar.set_description(
                ('{0}; it: {1}; phase: {2}; batch: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; x-reco {7:.3f};').format(\
                    batch_count+1, session.sample_i+1, session.phase, batch, session.alpha, reso, e, xr)
                )
            pbar.update(batch)
        else:
            ####### Critic update ########
            # real_cls = critic(real_z)
            real_cls    = critic(x,session.phase, session.alpha)

            wgan_C_loss = -torch.log(1.0-fake_cls).mean() - torch.log(real_cls).mean()
            # wgan_C_loss = -torch.log(1.0-fake_cls).mean() - torch.log(real_cls).mean()

            stats['real_cls'] = real_cls.mean().data.cpu().numpy()
            stats['fake_cls'] = fake_cls.mean().data.cpu().numpy()

            losses.append(wgan_C_loss)
            loss = sum(losses)
            stats['C_loss'] = loss.data.cpu().numpy()
            loss.backward()
            session.optimizerC.step()
            # torch.cuda.empty_cache()

        # del x, real_image, real_z, recon_x

        if args.use_TB:
            for key,val in stats.items():
                writer.add_scalar(key, val, session.sample_i)
            writer.add_scalar('LOD', session.phase + session.alpha, session.sample_i)
        elif batch_count % 100 == 0:
            print(stats)

        session.sample_i += batch
        batch_count += 1

        ########################  Saving ######################## 

        if batch_count % args.checkpoint_cycle == 0:
            for postfix in {'latest', str(session.sample_i).zfill(6)}:
                session.save_all('{}/{}_state'.format(args.checkpoint_dir, postfix))

            print("Checkpointed to {}".format(session.sample_i))

        ########################  Tests ######################## 

        evaluate.tests_run(g_running, encoder, test_data_loader, session,reconstruction = (batch_count % 100 == 0))


    pbar.close()

def setup():
    utils.make_dirs()
    if not args.testonly:
        config.log_args(args)

    if args.use_TB:
        from dateutil import tz
        from tensorboardX import SummaryWriter

        dt = datetime.now(tz.gettz('US/Pacific')).strftime(r"%y%m%d_%H%M")
        global writer
        writer = SummaryWriter("{}/{}/{}".format(args.summary_dir, args.save_dir, dt))

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)

def main():
    setup()
    session     = Session()
    checkname   = os.path.join(args.checkpoint_dir,'latest_state')

    if os.path.exists(checkname):
        session.load(checkname)
    else:
        print("Couldn't load previous session, create new!")
        session.create()

    print('PyTorch {}'.format(torch.__version__))

    if args.train_path:
        train_data_loader = data.get_loader(args.train_path)
    else:
        train_data_loader = None
    
    if args.test_path:
        test_data_loader = data.get_loader(args.test_path)
    elif args.aux_inpath:
        test_data_loader = data.get_loader(args.aux_inpath)
    else:
        test_data_loader = None

    # 4 modes: Train (with data/train), test (with data/test), aux-test (with custom aux_inpath), dump-training-set
    
    if args.run_mode == config.RUN_TRAIN:
        train(session, train_data_loader, test_data_loader, total_steps = args.total_kimg * 1000)

    elif args.run_mode == config.RUN_TEST:
        if args.reconstructions_N > 0 or args.interpolate_N > 0:
            evaluate.Utils.reconstruction_dryrun(session.generator, session.encoder, test_data_loader, session=session)
        evaluate.tests_run(session.generator, session.encoder, test_data_loader, session=session, writer=writer)
    elif args.run_mode == config.RUN_DUMP:
        session.phase = args.start_phase
        data.dump_training_set(train_data_loader, args.dump_trainingset_N, args.dump_trainingset_dir, session)

if __name__ == '__main__':
    main()


########## JUNK ########################

# import numpy as np
# from PIL import Image


# grad_loss   = get_grad_penalty(encoder,critic,
#                                x, fake_x, session.phase,
#                                batch, session.alpha)
# losses.append(grad_loss)


# def get_grad_penalty(encoder, critic, real_image, fake_image, step, batch, alpha):
#     eps = torch.rand(batch, 1, 1, 1).cuda()
#
#     x_hat = eps * real_image.data + (1 - eps) * fake_image.data
#     x_hat = Variable(x_hat, requires_grad=True)
#
#     hat_z = encoder(x_hat, step, alpha)
#     hat_predict = critic(hat_z)
#     grad_x_hat  = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
#
#     # Push the gradients of the interpolated samples towards 1
#     grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1)**2).mean()
#     return grad_penalty


# from torch.nn import functional as F


# one = torch.FloatTensor([1]).cuda(async=(args.gpu_count>1))

# if train_mode == config.MODE_GAN:
#
#     # Discriminator for real samples
#     real_z,real_predict = encoder(x, session.phase, session.alpha)
#     real_predict    = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
#     real_predict.backward(-one) # Towards 1
#
#     # (1) Generator => D. Identical to (2) see below
#
#     fake_predict, fake_image = D_prediction_of_G_output(generator, encoder, session.phase, session.alpha)
#     fake_predict.backward(one)
#
#     # Grad penalty
#
#     grad_penalty = get_grad_penalty(encoder, x, fake_image, session.phase, session.alpha)
#     grad_penalty.backward()
#
# elif train_mode == config.MODE_CYCLIC:


# class KLN01Loss(torch.nn.Module): #Adapted from https://github.com/DmitryUlyanov/AGE
#
#     def __init__(self, direction, minimize):
#         super(KLN01Loss, self).__init__()
#         self.minimize = minimize
#         assert direction in ['pq', 'qp'], 'direction?'
#
#         self.direction = direction
#
#     def forward(self, samples):
#
#         assert samples.nelement() == samples.size(1) * samples.size(0), '?'
#
#         samples = samples.view(samples.size(0), -1)
#
#         self.samples_var = utils.var(samples)
#         self.samples_mean = samples.mean(0)
#
#         samples_mean = self.samples_mean
#         samples_var = self.samples_var
#
#         if self.direction == 'pq':
#             t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
#             t2 = samples_var.log()
#
#             KL = (t1 + t2 - 0.5).mean()
#         else:
#             # In the AGE implementation, there is samples_var^2 instead of samples_var^1
#             t1 = (samples_var + samples_mean.pow(2)) / 2
#             # In the AGE implementation, this did not have the 0.5 scaling factor:
#             t2 = -0.5*samples_var.log()
#             KL = (t1 + t2 - 0.5).mean()
#
#         if not self.minimize:
#             KL *= -1
#
#         return KL


# if train_mode == config.MODE_CYCLIC:
#     if args.use_loss_z_reco:
#         stats['z_reconstruction_error'] = z_diff.data #[0]


# if args.use_real_x_KL:
#     # KL_real: - \Delta( e(X) , Z ) -> max_e
#     # KL_real = KL_minimizer(real_z) * args.real_x_KL_scale
#     e_losses.append(KL_real)
#
#     # stats['real_mean']  = KL_minimizer.samples_mean.data.mean()
#     # stats['real_var']   = KL_minimizer.samples_var.data.mean()
#     # stats['KL_real']    = KL_real.data #[0]
#     kls = "{0:.3f}".format(stats['KL_real'])

# zr, xr = (stats['z_reconstruction_error'], stats['x_reconstruction_error']) if train_mode == config.MODE_CYCLIC else (0.0, 0.0)
# e = (session.sample_i / float(epoch_len))
# pbar.set_description(
#     ('{0}; it: {1}; phase: {2}; b: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; KL(real/fake/fakeG): {7}; z-reco: {8:.2f}; x-reco {9:.3f}; real_var {10:.4f}').format(\
#         batch_count+1, session.sample_i+1, session.phase, b, session.alpha, reso, e, kls, zr, xr, stats['real_var'])
#     )


# if refresh_imagePool:
#     imagePoolSize = 200 if reso < 256 else 100
#     generatedImagePool = utils.ImagePool(imagePoolSize) #Reset the pool to avoid images of 2 different resolutions in the pool
#     refresh_imagePool = False
#     print('Image pool created with size {} because reso is {}'.format(imagePoolSize, reso))

####################### Training init #######################

# z = Variable( torch.FloatTensor(batch_size(reso), args.nz, 1, 1) ).cuda(async=(args.gpu_count>1))

# KL_minimizer = KLN01Loss(direction=args.KL, minimize=True)
# KL_maximizer = KLN01Loss(direction=args.KL, minimize=False)


# if train_mode == config.MODE_GAN:
#     fake_predict, _ = D_prediction_of_G_output(generator, encoder, session.phase, session.alpha)
#     loss = -fake_predict
#     g_losses.append(loss)
#
# elif train_mode == config.MODE_CYCLIC: #TODO We push the z variable around here like idiots
# def KL_of_encoded_G_output(generator, z):
#     utils.populate_z(z, args.nz, args.noise, batch_size(reso))
#     # z, label = utils.split_labels_out_of_latent(z)
#     fake = generator(z, session.phase, session.alpha)
#
#     egz = encoder(fake, session.phase, session.alpha)
#     # KL_fake: \Delta( e(g(Z)) , Z ) -> min_g
#     return egz, KL_minimizer(egz) * args.fake_G_KL_scale, z

# egz, kl, z = KL_of_encoded_G_output(generator, z)
#
# if args.use_loss_KL_z:
#     g_losses.append(kl) # G minimizes this KL
#     stats['KL(Phi(G))'] = kl.data #[0]
#     kls = "{0}/{1:.3f}".format(kls, stats['KL(Phi(G))'])
#
# if args.use_loss_z_reco:
#     # z = torch.cat((z, label), 1)
#     z_diff = utils.mismatch(egz, z, args.match_z_metric) * args.match_z # G tries to make the original z and encoded z match
#     g_losses.append(z_diff)


# if args.use_loss_fake_D_KL:
#     # TODO: The following codeblock is essentially the same as the KL_minimizer part on G side. Unify
#     utils.populate_z(z, args.nz, args.noise, batch_size(reso))
#     # z, label = utils.split_labels_out_of_latent(z)
#     fake = generator(z, session.phase, session.alpha).detach()
#
#     if session.alpha >= 1.0:
#         fake = generatedImagePool.query(fake.data)
#
#     # e(g(Z))
#     egz = encoder(fake, session.phase, session.alpha)
#
#     # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
#     # KL_fake = KL_maximizer(egz) * args.fake_D_KL_scale
#     # e_losses.append(KL_fake)
#
#     # stats['fake_mean'] = KL_maximizer.samples_mean.data.mean()
#     # stats['fake_var'] = KL_maximizer.samples_var.data.mean()
#     # stats['KL_fake'] = -KL_fake.data#[0]
#     kls = "{0}/{1:.3f}".format(kls, stats['KL_fake'])
#
#     if args.use_wpgan_grad_penalty:
#         grad_penalty = get_grad_penalty(encoder, x, fake, session.phase, session.alpha)


# # If we can switch from fade-training to stable-training
# if sample_i_current_stage >= args.images_per_stage:  # /2:
#     # if session.alpha < 1.0:
#     refresh_dataset = True  # refresh dataset generator since no longer have to fade
# #     match_x = args.match_x * args.matching_phase_x
# # else:
# match_x = args.match_x

# (f'{i + 1}; it: {iteration+1}; b: {b:.1f}; G: {gen_loss_val:.5f}; D: {disc_loss_val:.5f};'
# f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.3f}; Reso: {reso}; S-mean: {real_mean:.3f}; KL(real/fake/fakeG): {kls}; z-reco: {zr:.2f}'))


# benchmarking = False

# match_x = args.match_x

# To use labels, enable here and elsewhere:
    #label = Variable(torch.ones(batch_size_by_phase(step), args.n_label)).cuda()
    #               label = Variable(
    #                    torch.multinomial(
    #                        torch.ones(args.n_label), args.batch_size, replacement=True)).cuda()
