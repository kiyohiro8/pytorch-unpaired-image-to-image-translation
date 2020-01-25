
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ResidualBlock, SpectralNorm


class CycleGAN():
    def __init__(self, gen_num_channels, dis_num_channels):
        self.G_XY = Generator(gen_num_channels)
        self.G_YX = Generator(gen_num_channels)
        self.D_X = Discriminator(dis_num_channels)
        self.D_Y = Discriminator(dis_num_channels)


class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(3, num_channels, 7, bias=False),
                    nn.InstanceNorm2d(num_channels),
                    ResidualBlock(num_channels, norm_layer=nn.InstanceNorm2d),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = num_channels
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=2, bias=False),
                        nn.InstanceNorm2d(out_features),
                        ResidualBlock(out_features, norm_layer=nn.InstanceNorm2d),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(2):
            model += [ResidualBlock(in_features, norm_layer=nn.InstanceNorm2d)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):

            model += [  nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=1, bias=False),
                        nn.InstanceNorm2d(out_features),
                        ResidualBlock(out_features, norm_layer=nn.InstanceNorm2d),
                        nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, 3, 7, bias=False),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        model = [SpectralNorm(nn.Conv2d(3, num_channels, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(num_channels),
                 nn.LeakyReLU(0.2, inplace=True)]

        for _ in range(2):
            model += [SpectralNorm(nn.Conv2d(num_channels, num_channels*2, 4, stride=2, padding=1, bias=False)),
                      nn.InstanceNorm2d(num_channels*2),
                        nn.LeakyReLU(0.2, inplace=True)]
            num_channels *= 2

        # FCN classification layer
        model += [SpectralNorm(nn.Conv2d(num_channels, 1, 4, padding=1, bias=False))]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)