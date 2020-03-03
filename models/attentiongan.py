import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ResidualBlock, SpectralNorm, GlobalAveragePooling2d


class AttentionGuidedGAN():
    def __init__(self, num_channels):
        self.gen_XY = Generator(num_channels)
        self.gen_YX = Generator(num_channels)
        self.dis_X = Discriminator(num_channels)
        self.dis_Y = Discriminator(num_channels)
        self.attn_dis_X = Discriminator(num_channels)
        self.attn_dis_Y = Discriminator(num_channels)


class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        encoder = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(3, num_channels, 7, bias=False),
                    nn.InstanceNorm2d(num_channels),
                    ResidualBlock(num_channels, norm_layer=nn.InstanceNorm2d),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = num_channels
        out_features = in_features*2
        for _ in range(2):
            encoder += [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=2, bias=False),
                        nn.InstanceNorm2d(out_features),
                        ResidualBlock(out_features, norm_layer=nn.InstanceNorm2d),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(2):
            encoder += [ResidualBlock(in_features, norm_layer=nn.InstanceNorm2d)]

        self.encoder = nn.Sequential(*encoder)

        # Upsampling
        out_features = in_features//2
        image_decoder = []
        mask_decoder = []
        for _ in range(2):

            image_decoder += [  nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=1, bias=False),
                        nn.InstanceNorm2d(out_features),
                        ResidualBlock(out_features, norm_layer=nn.InstanceNorm2d),
                        nn.ReLU(inplace=True)]

            mask_decoder += [  nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=1, bias=False),
                        nn.InstanceNorm2d(out_features),
                        ResidualBlock(out_features, norm_layer=nn.InstanceNorm2d),
                        nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features//2

        # Output layer
        image_decoder += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, 3, 7, bias=False),
                    nn.Tanh() ]
        mask_decoder += [nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, 1, 7, bias=False),
                    nn.Sigmoid() ]

        self.image_decoder = nn.Sequential(*image_decoder)
        self.mask_decoder = nn.Sequential(*mask_decoder)

    def forward(self, x):
        z = self.encoder(x)
        out_image = self.image_decoder(z)
        out_mask = self.mask_decoder(z)
        return out_image, out_mask


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

class AttnDiscriminator(nn.Module):
    def __init__(self, num_channels):
        super(AttnDiscriminator, self).__init__()
        model = [SpectralNorm(nn.Conv2d(3, num_channels, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(num_channels),
                 nn.LeakyReLU(0.2, inplace=True)]

        for _ in range(2):
            model += [SpectralNorm(nn.Conv2d(num_channels, num_channels*2, 4, stride=2, padding=1, bias=False)),
                      nn.InstanceNorm2d(num_channels*2),
                        nn.LeakyReLU(0.2, inplace=True)]
            num_channels *= 2

        # FCN classification layer
        model += [SpectralNorm(nn.Conv2d(num_channels, 1, 4, padding=1, bias=False)),
                  GlobalAveragePooling2d()]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

        
