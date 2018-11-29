# import  torch
from    torch import nn
from    torch.nn import functional as F
import  utils
import  config
from    cnnutils import SpectralNormConv2d, EqualConv2d, PixelNorm

args   = config.get_config()

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding, kernel_size2=None, padding2=None,
                 pixel_norm=True, spectral_norm=False):

        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        if spectral_norm:
            # assert(0)
            self.conv = nn.Sequential(SpectralNormConv2d(in_channel, out_channel, kernel1, padding=pad1),
                                      nn.LeakyReLU(0.2),
                                      # nn.PReLU(),
                                      SpectralNormConv2d(out_channel, out_channel, kernel2, padding=pad2),
                                      # nn.PReLU())
                                      nn.LeakyReLU(0.2))

        else:
            if pixel_norm:
                # assert(0)
                self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                                          PixelNorm(),
                                          # nn.PReLU(),
                                          nn.LeakyReLU(0.2),
                                          EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                                          PixelNorm(),
                                          # nn.PReLU())
                                          nn.LeakyReLU(0.2))

            else:
                self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                                          # nn.PReLU(),
                                          nn.LeakyReLU(0.2),
                                          EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                                          nn.LeakyReLU(0.2))
                                          # nn.PReLU())

    def forward(self, input):
        out = self.conv(input)
        return out

gen_spectral_norm = False

class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()

        self.code_norm = PixelNorm()
        self.progression = nn.ModuleList([ConvBlock(nz, nz, 4, 3, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, int(nz/2), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/2), int(nz/4), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/4), int(nz/8), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/8), int(nz/16), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/16), int(nz/32), 3, 1, spectral_norm=gen_spectral_norm)])

        self.to_gray = nn.ModuleList([nn.Conv2d(nz, 1, 1), #Each has 1 channel and kernel size 1x1!
                                     nn.Conv2d(nz, 1, 1),
                                     nn.Conv2d(nz, 1, 1),
                                     nn.Conv2d(nz, 1, 1),
                                     nn.Conv2d(int(nz/2), 1, 1),
                                     nn.Conv2d(int(nz/4), 1, 1),
                                     nn.Conv2d(int(nz/8), 1, 1),
                                     nn.Conv2d(int(nz/16), 1, 1),
                                     nn.Conv2d(int(nz/32), 1, 1)])


    def forward(self, input, step=0, alpha=-1):
        # out_act = lambda x: x #nn.Tanh()
        # out = torch.cat([input, label], 1).unsqueeze(2).unsqueeze(3)
        out = input[...,None,None]

        for i, (conv, to_gray) in enumerate(zip(self.progression, self.to_gray)):

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2)
                out = conv(upsample)
            else:
                out = conv(out)

            if i == step: # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
                out = to_gray(out)

                if i > 0 and 0 <= alpha < 1:
                    # use previous 1x1 to gray transform
                    skip_gray = self.to_gray[i - 1](upsample)
                    out = (1 - alpha) * skip_gray + alpha * out
                break

        return out

pixelNormInDiscriminator = False

class Discriminator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        # self.binary_predictor = binary_predictor
        spectralNormInDiscriminator = True
        self.progression = nn.ModuleList([ConvBlock(int(nz/32), int(nz/16), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/16), int(nz/8), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/8), int(nz/4), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/4), int(nz/2), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/2), nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1, 4, 0,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator)]) #(nz+1 if use_mean_std_layer else nz)

        self.from_gray = nn.ModuleList([nn.Conv2d(1, int(nz/32), 1),
                                       nn.Conv2d(1, int(nz/16), 1),
                                       nn.Conv2d(1, int(nz/8), 1),
                                       nn.Conv2d(1, int(nz/4), 1),
                                       nn.Conv2d(1, int(nz/2), 1),
                                       nn.Conv2d(1, nz, 1),
                                       nn.Conv2d(1, nz, 1),
                                       nn.Conv2d(1, nz, 1),
                                       nn.Conv2d(1, nz, 1)])


        self.n_layer = len(self.progression)

    # c = 0
    def forward(self, input, step, alpha): #default was step=0, alpha=-1
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_gray[index](input)

            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool2d(out, 2)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_gray[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        z_out = out.squeeze(2).squeeze(2)
        # out = z_out.view(z_out.size(0), -1) #TODO: Is this needed?
        # return z_out
        return utils.normalize(z_out)


############# JUNK #########################

        # if self.binary_predictor:
        #     self.linear = nn.Linear(nz, 1 + n_label)


        # if self.binary_predictor:
        #     out = self.linear(z_out)
        #     return out[:, 0], out[:, 1:]
        # else:

        # print(input.size(), out.size(), step)

            # if i == 0 and use_mean_std_layer:
            #     mean_std = input.std(0).mean()
            #     mean_std = mean_std.expand(input.size(0), 1, 4, 4)
            #     out = torch.cat([out, mean_std], 1)

# def init_linear(linear):
#     init.xavier_normal(linear.weight)
#     linear.bias.data.zero_()
#
#
# def init_conv(conv, glu=True):
#     init.kaiming_normal(conv.weight)
#     if conv.bias is not None:
#         conv.bias.data.zero_()

            # if use_ALQ == 1:
            #     print('Reserved for future use')

        # self.label_embed = nn.Embedding(n_label, n_label)
        # self.label_embed.weight.data.normal_()


#        import ipdb
#        ipdb.set_trace()

# input = self.code_norm(input)
# ARI: THis causes internal assertion failure. Maybe we don't need the embedding?
# label = self.label_embed(label)


# self.to_rgb = nn.ModuleList([nn.Conv2d(nz, 3, 1), #Each has 3 out channels and kernel size 1x1!
        #                              nn.Conv2d(nz, 3, 1),
        #                              nn.Conv2d(nz, 3, 1),
        #                              nn.Conv2d(nz, 3, 1),
        #                              nn.Conv2d(int(nz/2), 3, 1),
        #                              nn.Conv2d(int(nz/4), 3, 1),
        #                              nn.Conv2d(int(nz/8), 3, 1),
        #                              nn.Conv2d(int(nz/16), 3, 1),
        #                              nn.Conv2d(int(nz/32), 3, 1)])

        # self.from_rgb = nn.ModuleList([nn.Conv2d(3, int(nz/32), 1),
        #                                nn.Conv2d(3, int(nz/16), 1),
        #                                nn.Conv2d(3, int(nz/8), 1),
        #                                nn.Conv2d(3, int(nz/4), 1),
        #                                nn.Conv2d(3, int(nz/2), 1),
        #                                nn.Conv2d(3, nz, 1),
        #                                nn.Conv2d(3, nz, 1),
        #                                nn.Conv2d(3, nz, 1),
        #                                nn.Conv2d(3, nz, 1)])
