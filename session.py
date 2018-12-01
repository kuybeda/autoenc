from    model import Generator, Encoder, Critic
from    torch import nn, optim
import  config
import  torch
import  os
import  copy

args   = config.get_config()

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

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

    return batch_table[reso]

class Session:
    def __init__(self):
        # Note: 4 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator
        # 4) You may need to warm up the g_running by running evaluate.reconstruction_dryrun() first

        self.alpha = -1
        self.sample_i = min(args.start_iteration, 0)
        self.phase = args.start_phase

        self.encoder    = nn.DataParallel(Encoder(nz=args.nz).cuda())
        self.generator  = nn.DataParallel(Generator(args.nz).cuda())
        self.g_running  = nn.DataParallel(Generator(args.nz).cuda())
        self.critic     = nn.DataParallel(Critic(args.nz).cuda())

        print("Using ", torch.cuda.device_count(), " GPUs!")

        self.reset_opt()

        print('Session created.')

    def cur_res(self):
        return  4 * 2 ** self.phase

    def cur_batch(self):
        return batch_size(self.cur_res())

    def reset_opt(self):
        self.optimizerG = optim.Adam(self.generator.parameters(), args.lr, betas=(0.0, 0.99))
        self.optimizerE = optim.Adam(self.encoder.parameters(), args.lr, betas=(0.0, 0.99))  # includes all the encoder parameters...
        self.optimizerC = optim.Adam(self.critic.parameters(), args.lr, betas=(0.0, 0.99))  # includes all the encoder parameters...

    def save_all(self, path):
        torch.save({'G_state_dict': self.generator.state_dict(),
                    'E_state_dict': self.encoder.state_dict(),
                    'C_state_dict': self.critic.state_dict(),
                    'G_running_state_dict': self.g_running.state_dict(),
                    'optimizerE': self.optimizerE.state_dict(),
                    'optimizerG': self.optimizerG.state_dict(),
                    'optimizerC': self.optimizerC.state_dict(),
                    'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha},
                   path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.sample_i = int(checkpoint['iteration'])

        self.generator.load_state_dict(checkpoint['G_state_dict'])
        self.g_running.load_state_dict(checkpoint['G_running_state_dict'])
        self.encoder.load_state_dict(checkpoint['E_state_dict'])
        self.critic.load_state_dict(checkpoint['C_state_dict'])

        if args.reset_optimizers <= 0:
            self.optimizerE.load_state_dict(checkpoint['optimizerE'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            self.optimizerC.load_state_dict(checkpoint['optimizerC'])
            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']
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
            if args.no_progression:
                self.sample_i = args.start_iteration = int((
                                                                       args.max_phase + 0.5) * args.images_per_stage)  # Start after the fade-in stage of the last iteration
                args.force_alpha = 1.0
                print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(args.start_iteration,
                                                                                                   args.force_alpha))
        else:
            reload_from = '{}/checkpoint/{}_state'.format(args.save_dir, str(args.start_iteration).zfill(
                6))  # e.g. '604000' #'600000' #latest'
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))

                if args.testonly:
                    self.generator = copy.deepcopy(self.g_running)
            else:
                assert (not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

        self.g_running.train(False)

        if args.force_alpha >= 0.0:
            self.alpha = args.force_alpha

        accumulate(self.g_running, self.generator, 0)

