import  torch
from    torch import nn
from    torch.nn import functional as F
from    cnnutils import EqualConv2d, DenseBlock, SpectralNormConv2d
from    torch.optim import Optimizer
# from    torch.autograd import Variable, grad

class OptimModule(nn.Module):
    ''' Module that returns and loads dictionary of all optimizers'''
    def __init__(self, load_optimizers=True):
        super().__init__()
        self.load_optimizers = load_optimizers

    def __get_all_optimizers(self):
        optimizers = {}
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for attr in members:
            m = getattr(self,attr)
            if isinstance(m,Optimizer):
                optimizers.update({attr:m})
        return optimizers

    def state_dict(self):
        ''' add optimizers state dict to module state dict '''
        sd = {'parent':super().state_dict()}
        for key,optim in self.__get_all_optimizers().items():
            sd.update({key:optim.state_dict()})
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # load parent
        super().load_state_dict(state_dict['parent'])
        if self.load_optimizers:
            for key,optim in self.__get_all_optimizers().items():
                optim.load_state_dict(state_dict[key])

N_DENSE_BLOCKS = 4

class Generator(nn.Module):
    def __init__(self, nz, pixel_norm=False, spectral_norm=True, use_shortcuts = False):
        super().__init__()
        self.use_shortcuts = use_shortcuts
        mxgrow = nz//N_DENSE_BLOCKS
        # prepare for concatenating with shortcuts
        short_fact = 2 if use_shortcuts else 1
        self.progression = nn.ModuleList([DenseBlock('block1', nz, mxgrow, nz//mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block2', short_fact*nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block3', short_fact*nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block4', short_fact*nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block5', short_fact*nz, mxgrow//2, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block6', short_fact*nz//2, mxgrow//4, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block7', short_fact*nz//4, mxgrow//8, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block8', short_fact*nz//8, mxgrow//8, nz // (2*mxgrow),
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block9', short_fact*nz//16, mxgrow//16, nz // (2*mxgrow),
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm)])

        self.to_gray = nn.ModuleList([EqualConv2d(nz, 1, 1), #Each has 1 channel and kernel size 1x1!
                                      EqualConv2d(nz, 1, 1),
                                      EqualConv2d(nz, 1, 1),
                                      EqualConv2d(nz, 1, 1),
                                      EqualConv2d(nz//2, 1, 1),
                                      EqualConv2d(nz//4, 1, 1),
                                      EqualConv2d(nz//8, 1, 1),
                                      EqualConv2d(nz//16, 1, 1),
                                      EqualConv2d(nz//32, 1, 1)])

        self.nonlin = nn.LeakyReLU(0.2)

    def forward(self, input, step=0, alpha=-1, shorts=None):
        out = input
        for i, (conv, to_gray) in enumerate(zip(self.progression, self.to_gray)):

            if i > 0 and step > 0:
                up_input = out
                upsample = F.interpolate(up_input, scale_factor=2, mode='nearest')
                # upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                incat = torch.cat((upsample,shorts[-i]),dim=1) if shorts is not None else upsample
                out = conv(incat)
            else:
                out = conv(out)

            if i == step: # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
                out = self.nonlin(out)
                out = to_gray(out)

                if i > 0 and 0 <= alpha < 1:
                    # use bilinear tranform here to match the bilinear interpolation in the input data
                    upsample    = F.interpolate(up_input, scale_factor=2, mode='bilinear', align_corners=False)
                    upsample    = self.nonlin(upsample)
                    skip_gray   = self.to_gray[i - 1](upsample)
                    out         = (1 - alpha) * skip_gray + alpha * out
                break
        return out

class Bottleneck(nn.Module):
    def __init__(self, nz, in_channels, pixel_norm, spectral_norm, make_shortcuts=False):
        super().__init__()
        self.make_shortcuts = make_shortcuts
        mxgrow = nz//N_DENSE_BLOCKS
        self.progression = nn.ModuleList([DenseBlock('block1', nz//32, mxgrow//8, nz // (2*mxgrow),
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block2', nz // 16, mxgrow//8, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block3', nz // 8, mxgrow//4, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block4', nz // 4, mxgrow//2, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block5', nz // 2, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block6', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block7', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block8', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block9', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm)])

        # network that acts on the narrowest point of the bottlneck
        self.narrowblocks   = nn.Sequential(DenseBlock('narblock1', nz, mxgrow, nz // mxgrow,
                                                        pixel_norm=pixel_norm, spectral_norm=spectral_norm,
                                                        dilations=[1, 2, 1, 2]),
                                            DenseBlock('narblock2', nz, mxgrow, nz // mxgrow,
                                                        pixel_norm=pixel_norm, spectral_norm=spectral_norm,
                                                        dilations=[1, 2, 1, 2]))

        self.from_gray      = nn.ModuleList([EqualConv2d(in_channels,nz // 32,1,1),
                                            EqualConv2d(in_channels, nz // 16, 1, 1),
                                            EqualConv2d(in_channels, nz // 8, 1, 1),
                                            EqualConv2d(in_channels, nz // 4, 1, 1),
                                            EqualConv2d(in_channels, nz // 2, 1, 1),
                                            EqualConv2d(in_channels, nz , 1, 1),
                                            EqualConv2d(in_channels, nz , 1, 1),
                                            EqualConv2d(in_channels, nz, 1, 1),
                                            EqualConv2d(in_channels, nz, 1, 1)])

        self.n_layer = len(self.progression)

    def forward(self, pool, input, phase, alpha):
        shorts = []
        for i in range(phase, -1, -1):
            index = self.n_layer - i - 1

            if i == phase:
                # first convolution that converts image to features
                out = self.from_gray[index](input)

            out = self.progression[index](out)
            if i == 0:
                # append narrow blocks here
                out = self.narrowblocks(out)
                # pass
            else:
                # update shortcuts
                shorts.append(out)
                out = F.avg_pool2d(out, 2)
                # implement smooth scale transition
                if i == phase and 0 <= alpha < 1:
                    skip    = F.avg_pool2d(input, 2)
                    skip    = self.from_gray[index + 1](skip)
                    out     = (1 - alpha) * skip + alpha * out

        if self.make_shortcuts:
            return out, shorts
        else:
            return out

class Encoder(Bottleneck):
    def __init__(self, nz, pixel_norm=True, spectral_norm=False, **kwargs):
        super().__init__(nz, 1, pixel_norm, spectral_norm, **kwargs)

    def forward(self,  input, step, alpha):
        out = super().forward(F.avg_pool2d, input, step, alpha)
        return out

class Critic(Bottleneck):
    def __init__(self, nz, pixel_norm=False, spectral_norm=True):
        super().__init__(nz, 2, pixel_norm, spectral_norm)
        self.classifier  = nn.Sequential(nn.LeakyReLU(0.2),
                                         SpectralNormConv2d(nz, nz, 3),
                                         nn.LeakyReLU(0.2),
                                         SpectralNormConv2d(nz, nz, 3),
                                         nn.LeakyReLU(0.2),
                                         nn.AdaptiveAvgPool2d(1),
                                         SpectralNormConv2d(nz, 1, 1,bias=False))

    def forward(self, input1, input2, step, alpha):
        # input = input1
        input = torch.cat((input1,input2),dim=1)
        out   = super().forward(F.avg_pool2d, input, step, alpha)
        # return torch.sigmoid(self.classifier(out)).squeeze(2).squeeze(2)
        return self.classifier(out).squeeze(2).squeeze(2)
