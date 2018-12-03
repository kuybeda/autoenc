from    model import Generator, Encoder, Critic
from    torch import nn, optim
import  config
import  torch
import  os
from    datetime import datetime
import  random
import  data
import  utils
from    torch.autograd import Variable #, grad
from    tqdm import tqdm

args   = config.get_config()

def batch_size(reso):
    if args.gpu_count == 1:
        save_memory = False
        if not save_memory:
            batch_table = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4, 1024: 1}
        else:
            batch_table = {4: 128, 8: 128, 16: 128, 32: 32, 64: 16, 128: 4, 256: 2, 512: 2, 1024: 1}
    elif args.gpu_count == 2:
        batch_table = {4: 256, 8: 256, 16: 256, 32: 128, 64: 64, 128: 32, 256: 16, 512: 8, 1024: 2}
    elif args.gpu_count == 4:
        batch_table = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32, 512: 16, 1024: 4}
    elif args.gpu_count == 8:
        batch_table = {4: 512, 8: 512, 16: 512, 32: 256, 64: 256, 128: 128, 256: 64, 512: 32, 1024: 8}
    else:
        assert (False)

    return batch_table[reso]*2

class Session:

    def __init__(self):
        # Note: 4 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator
        # 4) You may need to warm up the g_running by running evaluate.reconstruction_dryrun() first

        self.alpha      = -1
        self.kt         = 0.0
        self.sample_i   = min(args.start_iteration, 0)
        self.phase      = args.start_phase
        self.total_steps = args.total_kimg * 1000

        self.encoder    = nn.DataParallel(Encoder(args.nz).cuda())
        self.generator  = nn.DataParallel(Generator(args.nz).cuda())
        # self.attn       = nn.DataParallel(Attention(args.nz).cuda())
        # self.g_running  = nn.DataParallel(Generator(args.nz).cuda())
        self.critic     = nn.DataParallel(Critic(args.nz).cuda())
        print("Using ", torch.cuda.device_count(), " GPUs!")
        self.reset_opt()
        self.setup()
        print('Session created.')

    def setup(self):
        utils.make_dirs()
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

        if args.train_path:
            self.train_data_loader  = data.get_loader(args.train_path)
            self.epoch_len          = len(self.train_data_loader(1, 4).dataset)
        else:
            self.train_data_loader  = None
            self.epoch_len          = 0

        if args.test_path:
            self.test_data_loader = data.get_loader(args.test_path)
        elif args.aux_inpath:
            self.test_data_loader = data.get_loader(args.aux_inpath)
        else:
            self.test_data_loader = None

    def load_checkpoint(self):
        checkname = os.path.join(args.checkpoint_dir, 'latest_state')
        if os.path.exists(checkname):
            self.load(checkname)
        else:
            print("Couldn't load previous session, create new!")
            self.create()

    def cur_res(self):
        return  4 * 2 ** self.phase

    def cur_batch(self):
        return batch_size(self.cur_res())

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
        if args.step_offset != 0:
            if args.step_offset == -1:
                args.step_offset = self.sample_i
            print("Step offset is {}".format(args.step_offset))
            self.phase += args.phase_offset
            self.alpha = 0.0
            self.kt    = 0.0
        self.init_train_dataset()
        self.update_phase()

    def update_phase(self):
        steps_in_previous_phases    = max(self.phase * args.images_per_stage, args.step_offset)
        sample_i_current_stage      = self.sample_i - steps_in_previous_phases

        # If we can move to the next phase
        if sample_i_current_stage  >= args.images_per_stage:
            if self.phase < args.max_phase: # If any phases left
                iteration_levels = int(sample_i_current_stage / args.images_per_stage)
                self.phase += iteration_levels
                sample_i_current_stage -= iteration_levels * args.images_per_stage
                # reinitialize dataset to produce larger images
                self.init_train_dataset()
                print("iteration B alpha={} phase {} will be reduced to 1 and [max]".format(sample_i_current_stage, self.phase))
        # alpha growth 1/4th of the cycle
        self.alpha = min(1, sample_i_current_stage * 4.0 / args.images_per_stage)

    def init_train_dataset(self):
        batch, alpha, res, phase = self.cur_batch(), self.alpha, self.cur_res(), self.phase
        self.train_dataset = data.Utils.sample_data2(self.train_data_loader, batch, alpha, res, phase )


    def get_next_batch(self):
        try:
            real_image, _ = next(self.train_dataset)
        except (OSError, StopIteration):
            # restart dataset if epoch ended
            batch, alpha, res, phase = self.cur_batch(), self.alpha, self.cur_res(), self.phase
            train_dataset = data.Utils.sample_data2(self.train_data_loader, batch, alpha, res, phase)
            real_image, _ = next(train_dataset)
        return Variable(real_image).cuda(async=(args.gpu_count > 1))

    def handle_stats(self,stats):
        #  Display Statistics
        xr      = stats['x_err']
        zr      = stats['z_err']
        e       = (self.sample_i / float(self.epoch_len))
        batch   = self.cur_batch()
        self.pbar.set_description(
            ('{0}; it: {1}; phase: {2}; batch: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; x-err {7:.4f}; z-err {8:.4f};').format(\
                self.batch_count+1, self.sample_i+1, self.phase,
                batch, self.alpha, self.cur_res(), e, xr, zr)
            )
        self.pbar.update(batch)
        # Write data to Tensorboard #
        if args.use_TB:
            for key,val in stats.items():
                self.writer.add_scalar(key, val, self.sample_i)
            self.writer.add_scalar('LOD', self.phase + self.alpha, self.sample_i)
        elif self.batch_count % 100 == 0:
            print(stats)

    def maintain_phase(self):
        #######################  Phase Maintenance #######################
        self.sample_i += self.cur_batch()
        self.batch_count += 1
        self.update_phase()

    def save_checkpoint(self):
        if self.batch_count % args.checkpoint_cycle == 0:
            for postfix in {'latest', str(self.sample_i).zfill(6)}:
                self.save_all('{}/{}_state'.format(args.checkpoint_dir, postfix))
            print("\nCheckpointed to {}".format(self.sample_i))

    def finish(self):
        self.pbar.close()

    def reset_opt(self):
        self.optimizerG = optim.Adam(self.generator.parameters(), args.EGlr, betas=(0.0, 0.99))
        self.optimizerE = optim.Adam(self.encoder.parameters(), args.EGlr, betas=(0.0, 0.99))
        self.optimizerC = optim.Adam(self.critic.parameters(), args.Clr, betas=(0.0, 0.99))

    def save_all(self, path):
        torch.save({'G_state_dict': self.generator.state_dict(),
                    'E_state_dict': self.encoder.state_dict(),
                    'C_state_dict': self.critic.state_dict(),
                    # 'A_state_dict': self.attn.state_dict(),
                    # 'G_running_state_dict': self.g_running.state_dict(),
                    'optimizerE': self.optimizerE.state_dict(),
                    'optimizerG': self.optimizerG.state_dict(),
                    'optimizerC': self.optimizerC.state_dict(),
                    # 'optimizerA': self.optimizerA.state_dict(),
                    'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha,
                    'kt': self.kt},
                   path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.sample_i = int(checkpoint['iteration'])

        self.generator.load_state_dict(checkpoint['G_state_dict'])
        # self.g_running.load_state_dict(checkpoint['G_running_state_dict'])
        self.encoder.load_state_dict(checkpoint['E_state_dict'])
        self.critic.load_state_dict(checkpoint['C_state_dict'])
        # self.attn.load_state_dict(checkpoint['A_state_dict'])

        if not args.reset_optimizers:
            self.optimizerE.load_state_dict(checkpoint['optimizerE'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            self.optimizerC.load_state_dict(checkpoint['optimizerC'])
            # self.optimizerA.load_state_dict(checkpoint['optimizerA'])
            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']
        self.kt    = checkpoint['kt']
        self.phase = int(checkpoint['phase'])
        if args.start_phase > 0:  # If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            self.phase = min(args.start_phase, self.phase)
            print("Use start phase: {}".format(self.phase))
        if self.phase > args.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, args.max_phase))
            self.phase = args.max_phase

    def create(self):
        if args.start_iteration <= 0:
            args.start_iteration = 1
            # if args.no_progression:
            #     self.sample_i = args.start_iteration = int((args.max_phase + 0.5) * args.images_per_stage)  # Start after the fade-in stage of the last iteration
            #     args.force_alpha = 1.0
            #     print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(args.start_iteration,
            #                                                                                        args.force_alpha))
        else:
            reload_from = '{}/checkpoint/{}_state'.format(args.save_dir, str(args.start_iteration).zfill(
                6))  # e.g. '604000' #'600000' #latest'
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))

                # if args.testonly:
                #     self.generator = copy.deepcopy(self.g_running)
            else:
                assert (not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

########### JUNK ############################

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
