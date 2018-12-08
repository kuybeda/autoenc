from    model_utils import Generator, Encoder, Critic
import  config
from    torch import nn,optim
import  utils

args   = config.get_config()

class DualEncoder(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.rgb    = Encoder(nz)
        self.fir    = Encoder(nz)

class DualGenerator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.rgb    = Generator(nz)
        self.fir    = Generator(nz)

class DualCritic(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.rgb    = Critic(nz)
        self.fir    = Critic(nz)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        nz          = args.nz
        self.enc    = nn.DataParallel(DualEncoder(nz).cuda())
        self.gen    = nn.DataParallel(DualGenerator(nz).cuda())
        self.crt    = nn.DataParallel(DualCritic(nz).cuda())
        self.reset_optimization()

    def reset_optimization(self):
        self.optimE = optim.Adam(self.enc.parameters(), args.EGlr, betas=(0.0, 0.99))
        self.optimG = optim.Adam(self.gen.parameters(), args.EGlr, betas=(0.0, 0.99))
        self.optimC = optim.Adam(self.crt.parameters(), args.Clr, betas=(0.0, 0.99))

    def update(self, batch, phase, alpha):
        ##### Train Encoder and Generator
        utils.requires_grad(self.enc, True)
        utils.requires_grad(self.gen, True)
        utils.requires_grad(self.crt, False)
        self.enc.zero_grad()
        self.gen.zero_grad()

        rgb,fir = batch
        batch_size = rgb.shape[0]


