
import  torch
# from    torch.autograd import Variable
import  torchvision.utils

import  config
import  utils
# import  data
from    filetools import mkdir_assure
from    myplotlib import show_planes,imshow,clf

args = config.get_config()

def tests_run(session, reconstruction=True):
    if reconstruction:
        Utils.reconstruct_images(session)

    # if sampling:
    #     Utils.generate_random_samples(generator, session.sample_i, session)

class Utils:

    @staticmethod
    def reconstruct(x, session):
        encoder, generator = session.encoder, session.generator
        real_z = encoder(x, session.phase, session.alpha)
        fake_x = generator(real_z, session.phase, session.alpha)
        # fake_z = encoder(fake_x, session.phase, session.alpha)
        # mask   = attn(real_z, fake_z, session.phase, session.alpha)
        return fake_x.data

    @staticmethod
    def reconstruct_images(session):
        encoder,generator = session.encoder,session.generator
        batch, alpha, res, phase = session.cur_batch(), session.alpha, session.cur_res(), session.phase
        generator.eval()
        encoder.eval()

        utils.requires_grad(generator, False)
        utils.requires_grad(encoder, False)

        nsamples    = args.test_cols * args.test_rows
        session.test_data.init_epoch(nsamples, alpha, res, phase)
        input_ims   = session.get_next_test_batch() #test_data.next_batch()

        real_z      = encoder(input_ims, session.phase, session.alpha)
        reco_ims    = generator(real_z, session.phase, session.alpha).data

        # reco_ims    = Utils.reconstruct(input_ims, session)

        # join source and reconstructed images side by side
        out_ims     = torch.cat((input_ims,reco_ims), 1).view(2*nsamples,1,reco_ims.shape[-2],reco_ims.shape[-1])
        sample_dir  = '{}/recon'.format(args.save_dir)
        save_path   = '{}/{}.png'.format(sample_dir, session.sample_i + 1).zfill(6)
        mkdir_assure(sample_dir)

        print('\nSaving a new collage ...')
        torchvision.utils.save_image(out_ims, save_path, nrow=args.test_cols, normalize=True, padding=0, scale_each=False)
        # utils.requires_grad(generator, True)
        # utils.requires_grad(encoder, True)
        encoder.train()
        generator.train()

########### JUNK #################

    # @staticmethod
    # def reconstruction_dryrun(generator, encoder, loader, session):
    #     generator.eval()
    #     encoder.eval()
    #
    #     utils.requires_grad(generator, False)
    #     utils.requires_grad(encoder, False)
    #
    #     reso = session.cur_res()
    #
    #     warmup_rounds = 200
    #     print('Warm-up rounds: {}'.format(warmup_rounds))
    #
    #     if session.phase < 1:
    #         dataset = data.Utils.sample_data(loader, 4, reso)
    #     else:
    #         dataset = data.Utils.sample_data2(loader, 4, reso, session)
    #     real_image, _ = next(dataset)
    #     x = Variable(real_image).cuda()
    #
    #     for i in range(warmup_rounds):
    #         ex = encoder(x, session.phase, session.alpha, args.use_ALQ).detach()
    #         # ex, label = utils.split_labels_out_of_latent(ex)
    #         gex = generator(ex, session.phase, session.alpha).detach()
    #
    #     # utils.requires_grad(generator, True)
    #     # utils.requires_grad(encoder, True)
    #     encoder.train()
    #     generator.train()

        # dataset     = data.Utils.sample_data2(loader, nsamples, alpha, res, phase)
        # input_ims,_ = next(dataset)
        # input_ims   = session.get_next_test_batch(nsamples) #data.Utils.sample_data2(loader, nsamples, alpha, res, phase)


        # attn.eval()
        # utils.requires_grad(attn, False)
        # attn.train()

    # @staticmethod
    # def reconstruct(input_image, encoder, generator, session):
    #     ex  = encoder(input_image, session.phase, session.alpha)
    #     gex = generator(ex.detach(), session.phase, session.alpha).detach()
    #     return gex.data[:]

    # @staticmethod
    # def generate_random_samples(generator, global_i, session):
    #     generator.eval()
    #     utils.requires_grad(generator, False)
    #
    #     nsamples = args.test_cols * args.test_rows
    #
    #     # obtain samples
    #     myz = Variable(torch.randn(nsamples, args.nz)).cuda()
    #     # myz      = utils.normalize(myz)
    #
    #     # myz, input_class = utils.split_labels_out_of_latent(myz)
    #
    #     ims = generator(myz, session.phase, session.alpha).detach()  # .cpu().numpy()
    #
    #     sample_dir = '{}/sample'.format(args.save_dir)
    #     save_path = '{}/{}.png'.format(sample_dir, str(global_i + 1).zfill(6))
    #     mkdir_assure(sample_dir)
    #
    #     torchvision.utils.save_image(ims, save_path, nrow=args.test_cols, normalize=True, range=(-1, 1), padding=0)
    #
    #     utils.requires_grad(generator, True)
    #     generator.train()

    # reconstruction_set_x = None

# from    PIL import Image
# import  os
# import  numpy as np


    # interpolation_set_x = None
    #
    # @staticmethod
    # def interpolate_images(generator, encoder, loader, epoch, prefix, session, writer=None):
    #     generator.eval()
    #     encoder.eval()
    #
    #     utils.requires_grad(generator, False)
    #     utils.requires_grad(encoder, False)
    #
    #     nr_of_imgs = 4  # "Corners"
    #     reso = 4 * 2 ** session.phase
    #     if True:
    #         # if Utils.interpolation_set_x is None or Utils.interpolation_set_x.size(2) != reso or (phase >= 1 and alpha < 1.0):
    #         if session.phase < 1:
    #             dataset = data.Utils.sample_data(loader, nr_of_imgs, reso)
    #         else:
    #             dataset = data.Utils.sample_data2(loader, nr_of_imgs, reso, session)
    #         real_image, _ = next(dataset)
    #         Utils.interpolation_set_x = Variable(real_image, volatile=True).cuda()
    #
    #     latent_reso_hor = 8
    #     latent_reso_ver = 8
    #
    #     x = Utils.interpolation_set_x
    #
    #     z0 = encoder(Variable(x), session.phase, session.alpha, args.use_ALQ).detach()
    #
    #     t = torch.FloatTensor(latent_reso_hor * (latent_reso_ver + 1) + nr_of_imgs, x.size(1),
    #                           x.size(2), x.size(3))
    #     t[0:nr_of_imgs] = x.data[:]
    #
    #     special_dir = args.save_dir if not args.aux_outpath else args.aux_outpath
    #
    #     if not os.path.exists(special_dir):
    #         os.makedirs(special_dir)
    #
    #     for o_i in range(nr_of_imgs):
    #         single_save_path = '{}{}/interpolations_{}_{}_{}_orig_{}.png'.format(special_dir, prefix, session.phase,
    #                                                                              epoch, session.alpha, o_i)
    #         grid = torchvision.utils.save_image(x.data[o_i] / 2 + 0.5, single_save_path, nrow=1,
    #                                             padding=0)  # , normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?
    #
    #     # Origs on the first row here
    #     # Corners are: z0[0] ... z0[1]
    #     #                .
    #     #                .
    #     #              z0[2] ... z0[3]
    #
    #     delta_z_ver0 = ((z0[2] - z0[0]) / (latent_reso_ver - 1))
    #     delta_z_verN = ((z0[3] - z0[1]) / (latent_reso_ver - 1))
    #     for y_i in range(latent_reso_ver):
    #         if False:  # Linear interpolation
    #             z0_x0 = z0[0] + y_i * delta_z_ver0
    #             z0_xN = z0[1] + y_i * delta_z_verN
    #             delta_z_hor = (z0_xN - z0_x0) / (latent_reso_hor - 1)
    #             z0_x = Variable(torch.FloatTensor(latent_reso_hor, z0_x0.size(0)))
    #
    #             for x_i in range(latent_reso_hor):
    #                 z0_x[x_i] = z0_x0 + x_i * delta_z_hor
    #
    #         if True:  # Spherical
    #             t_y = float(y_i) / (latent_reso_ver - 1)
    #             # z0_y = Variable(torch.FloatTensor(latent_reso_ver, z0.size(0)))
    #             z0_y1 = Utils.slerp(z0[0].data, z0[2].data, t_y)
    #             z0_y2 = Utils.slerp(z0[1].data, z0[3].data, t_y)
    #             z0_x = Variable(torch.FloatTensor(latent_reso_hor, z0[0].size(0)))
    #             for x_i in range(latent_reso_hor):
    #                 t_x = float(x_i) / (latent_reso_hor - 1)
    #                 z0_x[x_i] = Utils.slerp(z0_y1, z0_y2, t_x)
    #
    #         # z0_x, label = utils.split_labels_out_of_latent(z0_x)
    #         gex = generator(z0_x, session.phase, session.alpha).detach()
    #
    #         # Recall that yi=0 is the original's row:
    #         t[(y_i + 1) * latent_reso_ver:(y_i + 2) * latent_reso_ver] = gex.data[:]
    #
    #         for x_i in range(latent_reso_hor):
    #             single_save_path = '{}{}/interpolations_{}_{}_{}_{}x{}.png'.format(special_dir, prefix, session.phase,
    #                                                                                epoch, session.alpha, y_i, x_i)
    #             grid = torchvision.utils.save_image(gex.data[x_i] / 2 + 0.5, single_save_path, nrow=1,
    #                                                 padding=0)  # , normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?
    #
    #     save_path = '{}{}/interpolations_{}_{}_{}.png'.format(special_dir, prefix, session.phase, epoch, session.alpha)
    #     grid = torchvision.utils.save_image(t / 2 + 0.5, save_path, nrow=latent_reso_ver,
    #                                         padding=0)  # , normalize=True) #range=(-1,1)) #, normalize=True) #, scale_each=True)?
    #     # Hacky but this is an easy way to rescale the images to nice big lego format:
    #     if session.phase < 4:
    #         im = Image.open(save_path)
    #         im2 = im.resize((1024, 1024))
    #         im2.save(save_path)
    #
    #     if writer:
    #         writer.add_image('interpolation_latest_{}'.format(session.phase), t / 2 + 0.5, session.phase)
    #
    #     generator.train()
    #     encoder.train()

    # @staticmethod
    # def slerp(p0, p1, t):
    #     omega = np.arccos(np.dot(p0,p1) / np.sqrt(np.dot(p0,p0)) / np.sqrt(np.dot(p1,p1)))
    #     k1 = np.sin((1-t) * omega) / np.sin(omega)
    #     k2 = np.sin(t * omega) / np.sin(omega)
    #     return k1*p0 + k2*p1


        # if individual_sampling and args.sample_N > 0:  # Full sample set generation.
        #     print("Full Test samples - generating...")
        #     Utils.generate_intermediate_samples(
        #         generator_for_testing,
        #         session.sample_i, session=session, collateImages=False)
        #     print("Full Test samples generated.")


        # if interpolation:
        #     for ii in range(args.interpolate_N):
        #         Utils.interpolate_images(generator_for_testing, encoder, test_data_loader, session.sample_i + ii,
        #                                  prefix='', session=session, writer=writer)

        # if not reconstructions or nr_of_imgs == 0: return
        # reso = session.cur_res()
        #
        # # First, create the single grid
        #
        # if Utils.reconstruction_set_x is None or Utils.reconstruction_set_x.size(2) != reso or (session.phase >= 1 and session.alpha < 1.0):
        #     if session.phase < 1:
        #         dataset = data.Utils.sample_data(loader, min(nr_of_imgs, 16), reso)
        #     else:
        #         dataset = data.Utils.sample_data2(loader, min(nr_of_imgs, 16), reso, session)
        #     Utils.reconstruction_set_x, _ = next(dataset)
        #
        # reco_image = Utils.reconstruct(Utils.reconstruction_set_x, encoder, generator, session)
        #
        # t = torch.FloatTensor(Utils.reconstruction_set_x.size(0) * 2, Utils.reconstruction_set_x.size(1),
        #                     Utils.reconstruction_set_x.size(2), Utils.reconstruction_set_x.size(3))
        #
        # t[0::2] = Utils.reconstruction_set_x[:]
        # t[1::2] = reco_image
        #
        # save_path = '{}{}/reconstructions_{}_{}_{}.png'.format(args.save_dir, prefix, session.phase, global_i, session.alpha)
        # grid = torchvision.utils.save_image(t[:nr_of_imgs] / 2 + 0.5, save_path, padding=0)
        #
        # # Hacky but this is an easy way to rescale the images to nice big lego format:
        # if session.phase < 4:
        #     h = np.ceil(nr_of_imgs / 8)
        #     h_scale = min(1.0, h/8.0)
        #     im = Image.open(save_path)
        #     im2 = im.resize((1024, int(1024 * h_scale)))
        #     im2.save(save_path)
        #
        # if writer:
        #     writer.add_image('reconstruction_latest_{}'.format(session.phase), t[:nr_of_imgs] / 2 + 0.5, session.phase)
        #
        # # Second, create the Individual images:
        # if session.phase < 1:
        #     dataset = data.Utils.sample_data(loader, 1, reso)
        # else:
        #     dataset = data.Utils.sample_data2(loader, 1, reso, session)
        #
        # special_dir = '{}/{}'.format(args.save_dir if not args.aux_outpath else args.aux_outpath, str(global_i).zfill(6))
        #
        # if not os.path.exists(special_dir):
        #     os.makedirs(special_dir)
        #
        # print("Save images: Alpha={}, phase={}, images={}, at {}".format(session.alpha, session.phase, nr_of_imgs, special_dir))
        # for o in range(nr_of_imgs):
        #     if o%500==0:
        #         print(o)
        #
        #     real_image, _ = next(dataset)
        #     reco_image = Utils.reconstruct(real_image, encoder, generator, session)
        #
        #     t = torch.FloatTensor(real_image.size(0) * 2, real_image.size(1),
        #                         real_image.size(2), real_image.size(3))
        #
        #     save_path_A = '{}/{}_orig.png'.format(special_dir, o)
        #     save_path_B = '{}/{}_pine.png'.format(special_dir, o)
        #
        #     torchvision.utils.save_image(real_image[0] / 2 + 0.5, save_path_A, padding=0)
        #     torchvision.utils.save_image(reco_image[0] / 2 + 0.5, save_path_B, padding=0)


        # # for outer_count in range(samplesRepeatN):
        #
        # colN = 1 if not collateImages else min(10, int(np.ceil(args.sample_N / 4.0)))
        # rowN = 128 if not collateImages else min(5, int(np.ceil(args.sample_N / 4.0)))
        # images = []
        # for _ in range(rowN):
        #     myz = Variable(torch.randn(args.n_label * colN, args.nz)).cuda()
        #     myz = utils.normalize(myz)
        #     myz, input_class = utils.split_labels_out_of_latent(myz)
        #
        #     new_imgs = generator(myz, input_class, session.phase, session.alpha).detach().data.cpu()
        #
        #     images.append(new_imgs)
        #
        # if collateImages:
        #     sample_dir = '{}/sample'.format(args.save_dir)
        #     if not os.path.exists(sample_dir):
        #         os.makedirs(sample_dir)
        #
        #     save_path = '{}/{}.png'.format(sample_dir, str(global_i + 1).zfill(6))
        #     torchvision.utils.save_image(
        #         torch.cat(images, 0),
        #         save_path,
        #         nrow=args.n_label * colN,
        #         normalize=True,
        #         range=(-1, 1),
        #         padding=0)
        #     # Hacky but this is an easy way to rescale the images to nice big lego format:
        #     im = Image.open(save_path)
        #     im2 = im.resize((1024, 512 if reso < 256 else 1024))
        #     im2.save(save_path)
        #
        #     if writer:
        #         writer.add_image('samples_latest_{}'.format(session.phase), torch.cat(images, 0), session.phase)
        # else:
        #     for ii, img in enumerate(images):
        #         torchvision.utils.save_image(
        #             img,
        #             '{}/{}_{}.png'.format(special_dir, str(global_i + 1).zfill(6), ii + outer_count * 128),
        #             nrow=args.n_label * colN,
        #             normalize=True,
        #             range=(-1, 1),
        #             padding=0)

        # if not collateImages:
        #     special_dir = '../metrics/{}/{}/{}'.format(args.data, reso, str(global_i).zfill(6))
        #     while os.path.exists(special_dir):
        #         special_dir += '_'
        #
        #     os.makedirs(special_dir)


# Total number is samplesRepeatN * colN * rowN
        # e.g. for 51200 samples, outcome is 5*80*128. Please only give multiples of 128 here.

        # samplesRepeatN = int(args.sample_N / 128) if not collateImages else 1
