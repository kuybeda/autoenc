from    torch import nn
import  config
import  torch
import  os
from    datetime import datetime
import  random
import  utils
from    tqdm import tqdm
from    filetools import mkdir_assure
import  torchvision.utils

args   = config.get_config()

class Session(nn.Module):

    def __init__(self):
        super().__init__()
        # Note: 4 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator
        # 4) You may need to warm up the g_running by running evaluate.reconstruction_dryrun() first

        self.alpha       = -1
        self.kt          = 0.0
        self.sample_i    = min(args.start_iteration, 0)
        self.phase       = args.start_phase
        self.total_steps = args.total_kimg * 1000
        # import a custom model
        Model            = getattr(__import__(args.modelmodule, fromlist=[None]),'Model')
        # self.add_module('model',Model())
        self.model       = Model()

        print("Using ", torch.cuda.device_count(), " GPUs!")
        self.setup()
        print('Session created.')

    def setup(self):
        utils.make_dirs()
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

        # import a custom data loader
        Datawrapper      = getattr(__import__(args.datamodule, fromlist=[None]),args.datawrapper)
        self.train_data  = Datawrapper(args.train_path)
        # self.epoch_len   = self.train_data.epoch_len()
        self.test_data   = Datawrapper(args.test_path)

    def cur_res(self):
        return  args.start_res * 2 ** self.phase

    def cur_batch(self):
        return self.model.batch_size(self.cur_res())

    def init_stats(self):
        if args.use_TB:
            from dateutil import tz
            from tensorboardX import SummaryWriter
            dt = datetime.now(tz.gettz('US/Pacific')).strftime(r"%y%m%d_%H%M")
            self.writer = SummaryWriter("{}/{}/{}".format(args.summary_dir, args.save_dir, dt))
        else:
            self.writer  = None
        # init progress bar
        self.pbar = tqdm(initial=self.sample_i, total=self.total_steps)

    def init_phase(self):
        # init phase params
        self.batch_count = 0
        self.init_data()
        self.update_phase()

    def update_model(self,batch):
        return self.model.update(batch,self.phase,self.alpha)

    def update_phase(self):
        steps_in_previous_phases    = self.phase * args.images_per_stage
        sample_i_current_stage      = self.sample_i - steps_in_previous_phases

        # If we can move to the next phase
        if sample_i_current_stage  >= args.images_per_stage:
            if self.phase < args.max_phase: # If any phases left
                iteration_levels = int(sample_i_current_stage / args.images_per_stage)
                self.phase += iteration_levels
                sample_i_current_stage -= iteration_levels * args.images_per_stage
                # reinitialize dataset to produce larger images
                self.init_data()
                self.model.reset_optimizers()
                print("\nStarting phase {}, resolution {}".format(self.phase, self.cur_res()))
        # alpha growth 1/4th of the cycle
        self.alpha = min(1, sample_i_current_stage * 4.0 / args.images_per_stage)

    def init_data(self):
        self.train_data.init_epoch(self.cur_batch(), self.cur_res())
        self.test_data.init_epoch(args.test_cols * args.test_rows, self.cur_res())

    def get_next_train_batch(self):
        return self.train_data.next_batch(self.alpha, self.phase)

    def get_next_test_batch(self):
        return self.test_data.next_batch(self.alpha, self.phase)

    def handle_stats(self,stats):
        # e           = (self.sample_i / float(self.epoch_len)/self.cur_batch())
        batch       = self.cur_batch()
        pbar_description = self.model.pbar_description(stats, batch, self.batch_count, self.sample_i,
                                                        self.phase, self.alpha, self.cur_res())
        self.pbar.set_description(pbar_description)
        self.pbar.update(batch)
        # Write data to Tensorboard #
        if args.use_TB:
            for key,val in stats.items():
                self.writer.add_scalar(key, val.data, self.sample_i)
            self.writer.add_scalar('LOD', self.phase + self.alpha, self.sample_i)
        elif self.batch_count % 100 == 0:
            print(stats)

    def maintain_phase(self):
        #######################  Phase Maintenance #######################
        self.sample_i += self.cur_batch()
        self.batch_count += 1
        self.update_phase()

    def finish(self):
        self.pbar.close()

    def save_checkpoint(self):
        if self.batch_count % args.checkpoint_cycle == 0:
            for postfix in {'latest', str(self.sample_i).zfill(6)}:
                path = '{}/{}_state'.format(args.checkpoint_dir, postfix)
                self.save(path)
                # torch.save({'session': self.state_dict()}, path)
            print("\nCheckpointed to {}".format(self.sample_i))

    def save(self, path):
        torch.save({'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha,
                    'kt': self.kt,
                    'model': self.model.state_dict()}, path)
        # torch.save({'session':self.state_dict()},path)

    def load_checkpoint(self):
        checkname = os.path.join(args.checkpoint_dir, 'latest_state')
        if os.path.exists(checkname):
            print("Loading previous session %s" % checkname)
            self.load(checkname)
        else:
            print("Couldn't load previous session, create new!")
            self.create()

    def load(self, path):
        checkpoint = torch.load(path)
        self.sample_i   = int(checkpoint['iteration'])
        self.phase      = int(checkpoint['phase'])
        self.alpha      = checkpoint['alpha']
        self.kt         = checkpoint['kt']
        self.model.load_state_dict(checkpoint['model'])

    def create(self):
        if args.start_iteration <= 0:
            args.start_iteration = 1
        else:
            reload_from = '{}/checkpoint/{}_state'.format(args.save_dir, str(args.start_iteration).zfill(
                6))  # e.g. '604000' #'600000' #latest'
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))
            else:
                assert (not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

    def write_tests(self):
        batch       = self.get_next_test_batch()
        out_ims     = self.model.dry_run(batch, self.phase, self.alpha)

        sample_dir  = '{}/recon'.format(args.save_dir)
        save_path   = '{}/{}.png'.format(sample_dir, self.sample_i + 1).zfill(6)

        mkdir_assure(sample_dir)
        print('\nSaving a new collage ...')
        torchvision.utils.save_image(out_ims, save_path, nrow=args.test_cols, normalize=True, padding=0, scale_each=True)


########### JUNK ############################

# from    model import Generator, Encoder, Critic
# from    torch import nn, optim
# import  data
# from    torch.autograd import Variable #, grad

#max(self.phase * args.images_per_stage, args.step_offset)

        # if args.step_offset != 0:
        #     if args.step_offset == -1:
        #         args.step_offset = self.sample_i
        #     print("Step offset is {}".format(args.step_offset))
        #     self.phase += args.phase_offset
        #     self.alpha = 0.0
        #     self.kt    = 0.0

        # nsamples = args.test_cols * args.test_rows
        # self.test_data.init_epoch(nsamples, self.alpha, self.cur_res(), self.phase)
        # self.test_data.stop_batches()

# ('{0}; it: {1}; phase: {2}; batch: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; x-err {7:.4f}; z-err {8:.4f};').format(\
#     self.batch_count+1, self.sample_i+1, self.phase, batch, self.alpha, self.cur_res(), e, xr, zr)
# )


# self.encoder    = nn.DataParallel(Encoder(args.nz).cuda())
        # self.generator  = nn.DataParallel(Generator(args.nz).cuda())
        # self.critic     = nn.DataParallel(Critic(args.nz).cuda())

        # self.sample_i = int(checkpoint['iteration'])
        #
        # self.generator.load_state_dict(checkpoint['G_state_dict'])
        # self.encoder.load_state_dict(checkpoint['E_state_dict'])
        # self.critic.load_state_dict(checkpoint['C_state_dict'])
        #
        # if not args.reset_optimizers:
        #     self.optimizerE.load_state_dict(checkpoint['optimizerE'])
        #     self.optimizerG.load_state_dict(checkpoint['optimizerG'])
        #     self.optimizerC.load_state_dict(checkpoint['optimizerC'])
        #     print("Reloaded old optimizers")
        # else:
        #     print("Despite loading the state, we reset the optimizers.")
        #
        # self.alpha = checkpoint['alpha']
        # self.kt    = checkpoint['kt']
        # self.phase = int(checkpoint['phase'])
        # if args.start_phase > 0:  # If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
        #     self.phase = min(args.start_phase, self.phase)
        #     print("Use start phase: {}".format(self.phase))
        # if self.phase > args.max_phase:
        #     print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, args.max_phase))
        #     self.phase = args.max_phase


        # torch.save({'iteration': self.sample_i,
        #             'phase': self.phase,
        #             'alpha': self.alpha,
        #             'kt': self.kt}, path)


    # def reset_opt(self):
    #     self.optimizerG = optim.Adam(self.generator.parameters(), args.EGlr, betas=(0.0, 0.99))
    #     self.optimizerE = optim.Adam(self.encoder.parameters(), args.EGlr, betas=(0.0, 0.99))
    #     self.optimizerC = optim.Adam(self.critic.parameters(), args.Clr, betas=(0.0, 0.99))


    # def get_next_test_batch(self):
    #     batch,self.test_dataset = self.get_next_batch(self.test_data, self.test_dataset)
    #     return batch

    # def get_next_batch(self, data, dataset):
    #     try:
    #         real_image, _ = next(dataset)
    #     except (OSError, StopIteration):
    #         # restart dataset if epoch ended
    #         alpha, res, phase   = self.alpha, self.cur_res(), self.phase
    #         dataset             = data.sample_data(batch_size, alpha, res, phase)
    #         real_image, _       = next(dataset)
    #     return Variable(real_image).cuda(async=(args.gpu_count > 1)), dataset
        # self.train_dataset = data.Utils.sample_data2(self.train_data_loader, batch, alpha, res, phase )
        # self.train_dataset = self.train_data.sample_data(batch, alpha, res, phase)

    # def init_test_dataset(self, batch_size):
    #     alpha, res, phase  = self.cur_batch(), self.alpha, self.cur_res(), self.phase
    #     self.test_dataset  = self.test_data.sample_data(batch_size, alpha, res, phase)


                # if args.testonly:
                #     self.generator = copy.deepcopy(self.g_running)

        # else:
        #     self.train_data_loader  = None
        #     self.epoch_len          = 0

         # if args.test_path:
        # elif args.aux_inpath:
        #     self.test_data_loader = data.get_loader(args.aux_inpath)
        # else:
        #     self.test_data_loader = None


            # if args.no_progression:
            #     self.sample_i = args.start_iteration = int((args.max_phase + 0.5) * args.images_per_stage)  # Start after the fade-in stage of the last iteration
            #     args.force_alpha = 1.0
            #     print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(args.start_iteration,
            #                                                                                        args.force_alpha))

        # self.g_running.train(False)
        # if args.force_alpha >= 0.0:
        #     self.alpha = args.force_alpha

        # accumulate(self.g_running, self.generator, 0)

        # if not args.testonly:
        #     config.log_args(args)

# def accumulate(model1, model2, decay=0.999):
#     par1 = dict(model1.named_parameters())
#     par2 = dict(model2.named_parameters())
#
#     for k in par1.keys():
#         par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)
