import  torch
from    torch import nn
from    torch.nn import functional as F
from    cnnutils import EqualConv2d, DenseBlock, SpectralNormConv2d
# from    torch.autograd import Variable, grad

class Generator(nn.Module):
    def __init__(self, nz, pixel_norm=False, spectral_norm=True):
        super().__init__()

        mxgrow = 16
        self.progression = nn.ModuleList([DenseBlock('block1', nz, mxgrow, nz//mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block2', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block3', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block4', nz, mxgrow, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block5', nz, mxgrow//2, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block6', nz//2, mxgrow//4, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block7', nz//4, mxgrow//8, nz // mxgrow,
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block8', nz//8, mxgrow//8, nz // (2*mxgrow),
                                                     pixel_norm=pixel_norm, spectral_norm=spectral_norm),
                                          DenseBlock('block9', nz//16, mxgrow//16, nz // (2*mxgrow),
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

    def forward(self, input, step=0, alpha=-1):
        out = input
        for i, (conv, to_gray) in enumerate(zip(self.progression, self.to_gray)):

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2, mode='nearest') #, align_corners=False)
                out = conv(upsample)
            else:
                out = conv(out)

            if i == step: # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
                out = self.nonlin(out)
                out = to_gray(out)

                if i > 0 and 0 <= alpha < 1:
                    # use previous 1x1 to gray transform
                    upsample    = self.nonlin(upsample)
                    skip_gray   = self.to_gray[i - 1](upsample)
                    out         = (1 - alpha) * skip_gray + alpha * out
                break
        return out

class Bottleneck(nn.Module):
    def __init__(self, nz, in_channels, pixel_norm, spectral_norm):
        super().__init__()
        mxgrow = 16
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

        self.from_gray = nn.ModuleList([EqualConv2d(in_channels,nz // 32,1,1),
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
        for i in range(phase, -1, -1):
            index = self.n_layer - i - 1

            if i == phase:
                out = self.from_gray[index](input)

            out = self.progression[index](out)
            if i > 0:
                out = F.avg_pool2d(out, 2)
                if i == phase and 0 <= alpha < 1:
                    skip     = F.avg_pool2d(input, 2)
                    skip    = self.from_gray[index + 1](skip)
                    out = (1 - alpha) * skip + alpha * out
        return out

class Encoder(Bottleneck):
    def __init__(self, nz, pixel_norm=True, spectral_norm=False):
        super().__init__(nz, 1, pixel_norm, spectral_norm)

    def forward(self,  input, step, alpha):
        out = super().forward(F.avg_pool2d, input, step, alpha)
        return out

class Critic(Bottleneck):
    def __init__(self, nz, pixel_norm=False, spectral_norm=True):
        super().__init__(nz, 2, pixel_norm, spectral_norm)
        self.classifier  = nn.Sequential(nn.LeakyReLU(0.2),
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
