import  torch
from    torch import nn
from    torch.autograd import Variable
from    math import sqrt
import  config
from    torch.nn import init

args   = config.get_config()

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda(async=(args.gpu_count>1))
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
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class SpectralNormConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)

        init.kaiming_normal_(conv.weight)
        conv.bias.data.zero_()
        self.conv = spectral_norm(conv)

    def forward(self, input):
        return self.conv(input)

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class Conv2dNorm(nn.Sequential):
    def __init__(self, name, in_channels, out_channels, kernel_size, padding,
                 pixel_norm=False, spectral_norm=False, bias = True):
        super().__init__()
        self.g_name = name
        if spectral_norm:
            self.add_module('conv', SpectralNormConv2d(in_channels, out_channels,
                                                       kernel_size=kernel_size, padding=padding, bias=bias))
        elif pixel_norm:
            self.add_module('conv', EqualConv2d(in_channels, out_channels,
                                                kernel_size=kernel_size, padding=padding, bias=bias))
            self.add_module('norm', PixelNorm())
        else:
            assert(0)
    def forward(self, x):
        return super().forward(x)

class DenseLayer(nn.Sequential):
    def __init__(self, name, in_channels, growth_rate, pixel_norm=False, spectral_norm=False, bias = True):
        super().__init__()
        self.g_name = name
        self.add_module('nonlin', nn.LeakyReLU(0.2))
        self.add_module('conv', Conv2dNorm('',in_channels, growth_rate, 3, padding=1, bias=bias,
                                            pixel_norm=pixel_norm, spectral_norm=spectral_norm))
    def forward(self, x):
        return super().forward(x)

class DenseBlock(nn.Module):
    def __init__(self, name, in_channels, growth_rate, n_layers, fixed_channels=True, bias=True,
                 pixel_norm=False, spectral_norm=False):
        super().__init__()
        self.g_name = name
        self.fixed_channels = fixed_channels
        self.layers = nn.ModuleList([DenseLayer(name + 'denselayer_%d/' % i, in_channels + i*growth_rate, growth_rate,
                                                bias=bias, pixel_norm=pixel_norm, spectral_norm=spectral_norm) for i in range(n_layers)])
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
            assert(0)
            # here the number of channels grows int + nblocks*growth_rate
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding, kernel_size2=None, padding2=None,
                 pixel_norm=False, spectral_norm=False):

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
                                      SpectralNormConv2d(out_channel, out_channel, kernel2, padding=pad2),
                                      nn.LeakyReLU(0.2))

        elif pixel_norm:
                # assert(0)
                self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                                          PixelNorm(),
                                          nn.LeakyReLU(0.2),
                                          EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                                          PixelNorm(),
                                          nn.LeakyReLU(0.2))

        else:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
                                      nn.LeakyReLU(0.2),
                                      EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                                      nn.LeakyReLU(0.2))
                                      # nn.PReLU())

    def forward(self, input):
        return self.conv(input)



