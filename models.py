import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import SpectralNorm, ResidualBlock, GlobalAveragePooling2d


class CycleGAN():
    def __init__(self, gen_num_channels, dis_num_channels):
        self.G_XY = self._get_generator(gen_num_channels)
        self.G_YX = self._get_generator(gen_num_channels)
        self.D_X = self._get_discriminator(dis_num_channels)
        self.D_Y = self._get_discriminator(dis_num_channels)

    def _get_generator(self, num_channels):
        model = [   nn.ReplicationPad2d(3),
                    nn.Conv2d(3, num_channels, 7, bias=False),
                    nn.InstanceNorm2d(num_channels),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = num_channels
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.ReplicationPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=2, bias=False),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(6):
            model += [ResidualBlock(in_features, norm_layer=nn.InstanceNorm2d)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):

            model += [  nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.ReplicationPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=1, bias=False),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReplicationPad2d(3),
                    nn.Conv2d(in_features, 3, 7, bias=False),
                    nn.Tanh() ]

        return nn.Sequential(*model)

    def _get_discriminator(self, num_channels):

        model = [nn.Conv2d(3, num_channels, 4, stride=2, bias=False),
                 nn.InstanceNorm2d(num_channels),
                 nn.LeakyReLU(0.2, inplace=True)]

        for _ in range(3):
            model += [nn.Conv2d(num_channels, num_channels*2, 4, stride=2, bias=False),
                      nn.InstanceNorm2d(num_channels*2),
                        nn.LeakyReLU(0.2, inplace=True)]
            num_channels *= 2

        # FCN classification layer
        model += [nn.Conv2d(num_channels, 1, 4, padding=1, bias=False),
                  GlobalAveragePooling2d(),
                  nn.Sigmoid()]

        return nn.Sequential(*model)


class MUNIT():
    def __init__(self):
        self.Encoder_X = Encoder()
        self.Encoder_Y = Encoder()
        self.Decoder_X = Decoder()
        self.Decoder_Y = Decoder()
        self.Discriminator_X = Discriminator()
        self.Discriminator_Y = Discriminator()

    class Encoder(nn.Module):
        def __init__(self):
            pass

        def forward(self, image):
            pass

    class Decoder(nn.Module):
        def __init__(self):
            pass

        def forward(self, content, style):
            pass

    class Discriminator(nn.Module):
        def __init__(self):
            pass

        def forward(self, image):
            pass


