import  torch
from    torch import nn
from    math import sqrt
from    torch.nn import init
from    torch.autograd import Variable, grad
import  numpy as np
# import  utils
# import  config

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            # u = u.cuda(async=(args.gpu_count>1))
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)
    return module

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

#TODO For fp16, NVIDIA puts G.pixelnorm_epsilon=1e-4
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # (input.norm(2, dim=1, keepdim=True) + 1e-8)  #
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class SpectralNormConv(nn.Module):
    def __init__(self, *args, bias=True, conv=nn.Conv2d, **kwargs):
        super().__init__()
        convfun = conv(*args, bias=bias, **kwargs)
        # if const_weight_init:
        #     init.constant_(convfun.weight,1.0) #1.0/np.prod(conv.weight.shape[1:]
        # else:
        init.kaiming_normal_(convfun.weight)
        if bias:
            convfun.bias.data.zero_()
        self.conv   = spectral_norm(convfun)

    def forward(self, input):
        out = self.conv(input)
        return out

class EqualConv(nn.Module):
    def __init__(self, *args, bias=True, conv=nn.Conv2d, **kwargs):
        super().__init__()

        conv = conv(*args, **kwargs)
        conv.weight.data.normal_()
        if bias:
            conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class ConvNorm(nn.Sequential):
    def __init__(self, name, in_channels, out_channels, kernel_size, padding=0,
                 pixel_norm=False, spectral_norm=False, bias = True, **kwargs):
        super().__init__()
        self.g_name = name
        if spectral_norm:
            self.add_module('conv', SpectralNormConv(in_channels, out_channels,
                                                       kernel_size=kernel_size,
                                                       padding=padding, bias=bias, **kwargs))
        else:
            self.add_module('conv', EqualConv(in_channels, out_channels,
                                                kernel_size=kernel_size,
                                                padding=padding, bias=bias, **kwargs))
        if pixel_norm:
            self.add_module('norm', PixelNorm())

    def forward(self, x):
        return super().forward(x)

class DenseLayer(nn.Sequential):
    def __init__(self, name, in_channels, growth_rate, pixel_norm=False,
                 spectral_norm=False, bias = True, dilation=1,
                 nonlin=nn.LeakyReLU(0.2), batchnorm=False,**kwargs):
        super().__init__()
        self.g_name = name
        if batchnorm:
            self.add_module('batchnorm', nn.BatchNorm2d(in_channels)),
        self.add_module('nonlin', nonlin)
        self.add_module('conv', ConvNorm('',in_channels, growth_rate, 3, padding=dilation, bias=bias,
                                         pixel_norm=pixel_norm, spectral_norm=spectral_norm,
                                         dilation=dilation, **kwargs))
    def forward(self, x):
        return super().forward(x)

class DenseBlock(nn.Module):
    def __init__(self, name, in_channels, growth_rate, n_layers, fixed_channels=True, bias=True,
                 pixel_norm=False, spectral_norm=False, dilations=[1], nonlin=nn.LeakyReLU(0.2), **kwargs):
        super().__init__()
        self.g_name = name
        self.fixed_channels = fixed_channels
        self.layers = nn.ModuleList([DenseLayer(name + 'denselayer_%d/' % i, in_channels + i*growth_rate, growth_rate,
                                                bias=bias, pixel_norm=pixel_norm, spectral_norm=spectral_norm,
                                                dilation=dilations[min(i,len(dilations)-1)], nonlin=nonlin, **kwargs) \
                                     for i in range(n_layers)])
    def forward(self, x):
        if self.fixed_channels:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            # assert(0)
            # here the number of channels grows int + nblocks*growth_rate
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


########### JUNK ########################################
# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size,
#                  padding, kernel_size2=None, padding2=None,
#                  pixel_norm=False, spectral_norm=False):
#
#         super().__init__()
#
#         pad1 = padding
#         pad2 = padding
#         if padding2 is not None:
#             pad2 = padding2
#
#         kernel1 = kernel_size
#         kernel2 = kernel_size
#         if kernel_size2 is not None:
#             kernel2 = kernel_size2
#
#         if spectral_norm:
#             self.conv = nn.Sequential(SpectralNormConv2d(in_channel, out_channel, kernel1, padding=pad1),
#                                       nn.LeakyReLU(0.2),
#                                       SpectralNormConv2d(out_channel, out_channel, kernel2, padding=pad2),
#                                       nn.LeakyReLU(0.2))
#
#         elif pixel_norm:
#             self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
#                                       PixelNorm(),
#                                       nn.LeakyReLU(0.2),
#                                       EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
#                                       PixelNorm(),
#                                       nn.LeakyReLU(0.2))
#
#         else:
#             self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
#                                       nn.LeakyReLU(0.2),
#                                       EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
#                                       nn.LeakyReLU(0.2))
#                                       # nn.PReLU())
#
#     def forward(self, input):
#         return self.conv(input)



