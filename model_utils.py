import  torch
from    torch import nn
from    torch.nn import functional as F
from    cnnutils import EqualConv, ConvNorm, DenseBlock, SpectralNormConv
from    torch.optim import Optimizer
# from    torch.autograd import Variable, grad

def real_fake_loss(crt_real, crt_fake):
    return -(crt_fake * (crt_real.detach() > crt_fake.detach()).float()).mean()
    # return -crt_fake.mean()


def crt_loss_balanced(cr,cf):
    return (cf - cr + torch.abs(cf + cr)).mean()

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

class Generator(nn.Module):
    def __init__(self, out_channels, nz, ngrow, nblocks,
                 pixel_norm=False, spectral_norm=False, use_shortcuts = False,
                 nonlin=nn.LeakyReLU(0.2), **kwargs):

        super().__init__()
        assert(len(ngrow) == len(nblocks))
        self.use_shortcuts = use_shortcuts
        self.n_layers = len(ngrow)
        # prepare for concatenating with shortcuts
        short_fact = 2 if use_shortcuts else 1
        prog = []
        gray = []
        nin  = nz
        for l in range(self.n_layers):
            prog.append(DenseBlock('block%d' % l, nin, ngrow[l], nblocks[l],
                                   pixel_norm=pixel_norm, spectral_norm=spectral_norm,
                                   nonlin=nonlin,**kwargs))
            ngray = ngrow[l]*nblocks[l]
            nin   = short_fact*ngray
            gray.append(ConvNorm('conv%d' % l, ngray, out_channels, kernel_size=1, **kwargs))

        self.progression  = nn.ModuleList(prog)
        self.to_gray      = nn.ModuleList(gray)
        self.nonlin       = nonlin

    def forward(self, input, shorts=None, phase=None, alpha=1.0):
        phase = self.n_layers if phase is None else phase
        out   = input
        linmode = 'bilinear' if input.dim() < 5 else 'trilinear'
        for i, (conv, to_gray) in enumerate(zip(self.progression, self.to_gray)):
            if i > 0 and phase > 0:
                up_input = out
                upsample    = F.interpolate(up_input, scale_factor=2, mode='nearest')
                incat       = torch.cat((upsample,shorts[-i]),dim=1) if shorts is not None else upsample
                out         = conv(incat)
                # make grayscale resnet link
                skipgray    = to_gray(self.nonlin(out))
                # use bilinear tranform here to match the bilinear interpolation in the input data
                skipout     = F.interpolate(skipout, scale_factor=2, mode=linmode, align_corners=False)
                skipout = (1 - 0.5*alpha) * skipout + 0.5*alpha * skipgray  if i == phase else 0.5*(skipgray + skipout)
            else:
                out     = conv(out)
                skipout = to_gray(self.nonlin(out))
        return skipout

class Bottleneck(nn.Module):
    def __init__(self, in_channels, nfirst_channels, ngrow, nblocks,
                 pixel_norm=False, spectral_norm=False, nonlin=nn.LeakyReLU(0.2),
                 make_shortcuts=False, **kwargs):

        super().__init__()
        self.make_shortcuts = make_shortcuts
        assert(len(ngrow) == len(nblocks))
        self.n_layers = len(ngrow)
        # create progression modules
        prog   = []
        gray   = []
        nout   = nfirst_channels
        for l in range(self.n_layers):
            prog.append(DenseBlock('block%d'%l, nout, ngrow[l], nblocks[l],
                                   pixel_norm=pixel_norm, spectral_norm=spectral_norm,
                                   nonlin=nonlin, **kwargs))
            gray.append(ConvNorm('conv%d'%l, in_channels, nout, kernel_size=1, pixel_norm=pixel_norm, spectral_norm=spectral_norm, **kwargs))
            nout = ngrow[l]*nblocks[l]

        gray.append(ConvNorm('conv%d' % l, in_channels, nout, kernel_size=1,
                             pixel_norm=pixel_norm, spectral_norm=spectral_norm, **kwargs))

        self.progression  = nn.ModuleList(prog)
        self.narrowblocks = nn.Sequential()
        self.from_gray    = nn.ModuleList(gray)

    def forward(self, input, phase=None, alpha=1.0):
        phase  = self.n_layers if phase is None else phase
        pool   = F.avg_pool2d if input.dim() < 5 else F.avg_pool3d

        shorts = []
        out    = self.from_gray[0](input)
        skipin = input

        for i in range(phase):
            out = self.progression[i](out)
            # update shortcuts
            shorts.append(out)
            out = pool(out, 2)
            # implement smooth scale transition
            skipin  = pool(skipin, 2)
            skip    = self.from_gray[i+1](skipin)
            out = (1 - 0.5*alpha) * skip + 0.5*alpha * out if i==0 else 0.5*(skip + out)

        # apply bottleneck blocks here
        out = self.narrowblocks(out)

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
    def __init__(self, n_in, nz, pixel_norm=False, spectral_norm=True):
        super().__init__(nz, n_in, pixel_norm, spectral_norm)
        self.classifier  = nn.Sequential(nn.LeakyReLU(0.2),
                                         SpectralNormConv(nz, nz, 3),
                                         nn.LeakyReLU(0.2),
                                         SpectralNormConv(nz, nz, 3),
                                         nn.LeakyReLU(0.2),
                                         nn.AdaptiveAvgPool2d(1),
                                         SpectralNormConv(nz, 1, 1,bias=False))

    def forward(self, input, step, alpha):
        # input = input1
        # input = torch.cat((input1,input2),dim=1)
        out   = super().forward(F.avg_pool2d, input, step, alpha)
        # return torch.sigmoid(self.classifier(out)).squeeze(2).squeeze(2)
        return self.classifier(out).squeeze(2).squeeze(2)


################ JUNK #############################

# make side branch for graylevel pyramid
# upgray      = F.interpolate(up_input, scale_factor=2, mode='bilinear', align_corners=False)
# skip_gray   = self.to_gray[i - 1](self.nonlin(upgray))


# if i == step: # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
#     # out = self.nonlin(out)
#     # out = to_gray(out)
#     # if i > 0 and 0 <= alpha < 1:
#     #     # use bilinear tranform here to match the bilinear interpolation in the input data
#     #     upsample    = F.interpolate(up_input, scale_factor=2, mode='bilinear', align_corners=False)
#     #     upsample    = self.nonlin(upsample)
#     #     skip_gray   = self.to_gray[i - 1](upsample)
#     #     out         = (1 - alpha) * skip_gray + alpha * out
#     break


# self.to_gray = nn.ModuleList([ConvNorm(nz, 1, 1), #Each has 1 channel and kernel size 1x1!
        #                               EqualConv(nz, 1, 1),
        #                               EqualConv(nz, 1, 1),
        #                               EqualConv(nz, 1, 1),
        #                               EqualConv(nz//2, 1, 1),
        #                               EqualConv(nz//4, 1, 1),
        #                               EqualConv(nz//8, 1, 1),
        #                               EqualConv(nz//16, 1, 1),
        #                               EqualConv(nz//32, 1, 1)])

        # self.progression = nn.ModuleList([DenseBlock('block1', nz, mxgrow, nz//mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block2', short_fact*nz, mxgrow, nz // mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block3', short_fact*nz, mxgrow, nz // mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block4', short_fact*nz, mxgrow, nz // mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block5', short_fact*nz, mxgrow//2, nz // mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block6', short_fact*nz//2, mxgrow//4, nz // mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block7', short_fact*nz//4, mxgrow//8, nz // mxgrow,
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block8', short_fact*nz//8, mxgrow//8, nz // (2*mxgrow),
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
        #                                   DenseBlock('block9', short_fact*nz//16, mxgrow//16, nz // (2*mxgrow),
        #                                              pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs)])



        # self.from_gray      = nn.ModuleList([EqualConv(in_channels,nz // 32,1,1),
        #                                     EqualConv(in_channels, nz // 16, 1, 1),
        #                                     EqualConv(in_channels, nz // 8, 1, 1),
        #                                     EqualConv(in_channels, nz // 4, 1, 1),
        #                                     EqualConv(in_channels, nz // 2, 1, 1),
        #                                     EqualConv(in_channels, nz , 1, 1),
        #                                     EqualConv(in_channels, nz , 1, 1),
        #                                     EqualConv(in_channels, nz, 1, 1),
        #                                     EqualConv(in_channels, nz, 1, 1)])

        # mxgrow = nz // n_blocks
        # nn.ModuleList([DenseBlock('block1', nz//32, mxgrow//8, nz // (2*mxgrow),
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block2', nz // 16, mxgrow//8, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block3', nz // 8, mxgrow//4, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block4', nz // 4, mxgrow//2, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block5', nz // 2, mxgrow, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block6', nz, mxgrow, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block7', nz, mxgrow, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block8', nz, mxgrow, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs),
            #                               DenseBlock('block9', nz, mxgrow, nz // mxgrow,
            #                                          pixel_norm=pixel_norm, spectral_norm=spectral_norm,**kwargs)])

        # # network that acts on the narrowest point of the bottlneck
        # self.narrowblocks   = nn.Sequential(DenseBlock('narblock1', nz, mxgrow, nz // mxgrow,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm,
        #                                                 dilations=[1, 2, 1, 2]),
        #                                     DenseBlock('narblock2', nz, mxgrow, nz // mxgrow,
        #                                                 pixel_norm=pixel_norm, spectral_norm=spectral_norm,
        #                                                 dilations=[1, 2, 1, 2]))

