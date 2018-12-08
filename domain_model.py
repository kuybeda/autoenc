from    torch import nn

class Model(nn.Module):
    class Bottleneck(nn.Module):
        def __init__(self, nz, in_channels, pixel_norm, spectral_norm):
            super().__init__()
