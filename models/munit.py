
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import ResidualBlock, MLP, SpectralNorm, LayerNorm, AdaptiveInstanceNorm2d


class MUNIT():
    def __init__(self, num_channels, style_dim, num_downsampling=2, num_res_blocks=4):
        self.style_enc_X = StyleEncoder(num_channels, style_dim)
        self.style_enc_Y = StyleEncoder(num_channels, style_dim)
        self.content_enc_X = ContentEncoder(num_channels, num_downsampling, num_res_blocks)
        self.content_enc_Y = ContentEncoder(num_channels, num_downsampling, num_res_blocks)
        self.dec_X = Decoder(num_channels, style_dim, num_downsampling, num_res_blocks)
        self.dec_Y = Decoder(num_channels, style_dim, num_downsampling, num_res_blocks)
        self.dis_X = Discriminator(num_channels)
        self.dis_Y = Discriminator(num_channels)

class StyleEncoder(nn.Module):
    def __init__(self, num_channels, style_dim):
        super(StyleEncoder, self).__init__()
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(3, num_channels, 7, bias=False),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = num_channels
        out_features = in_features*2
        for i in range(4):
            model += [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 4, stride=2, bias=False),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            if i >= 2:
                out_features = in_features
            else: 
                out_features = in_features*2
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(out_features, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*model)
    
    def forward(self, image):
        return self.model(image)


class ContentEncoder(nn.Module):
    def __init__(self, num_channels, num_downsampling, num_res_blocks):
        super(ContentEncoder, self).__init__()
        model = [   nn.ReflectionPad2d(3),
        nn.Conv2d(3, num_channels, 7, bias=False),
        nn.ReLU(inplace=True) ]
        in_features = num_channels
        out_features = in_features * 2

        for _ in range(num_downsampling):
            model += [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 4, stride=2, bias=False),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2
        
        for _ in range(num_res_blocks):
            model += [ResidualBlock(in_features, norm_layer=nn.InstanceNorm2d)]

        self.model = nn.Sequential(*model)
    
    def forward(self, image):
        return self.model(image)


class Decoder(nn.Module):
    def __init__(self, num_channels, style_dim, num_downsampling, num_res_blocks):
        super(Decoder, self).__init__()
        num_channels = num_channels * 2 ** num_downsampling
        self.mlp = MLP(style_dim, num_channels*2, 256)
        decoder_res = []
        decoder_upsample = []
        for _ in range(num_res_blocks):
            decoder_res += [ResidualBlock(num_channels, AdaptiveInstanceNorm2d)]
        for _ in range(num_downsampling):
            decoder_upsample +=[nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.ReflectionPad2d(2),
                        nn.Conv2d(num_channels, num_channels // 2, 5, 1),
                        LayerNorm(num_channels // 2),
                        nn.ReLU(inplace=True)]
            num_channels = num_channels // 2
        decoder_upsample += [nn.ReflectionPad2d(3),
                    nn.Conv2d(num_channels, 3, kernel_size=7, stride=1),
                    nn.Tanh()]
        self.res = nn.Sequential(*decoder_res)
        self.upsample = nn.Sequential(*decoder_upsample)
    
    def _assign_adain_params(self, adain_params):
        # assign the adain_params to the AdaIN layers in model
        for m in self.res.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, :m.num_features]
                m.weight = adain_params[:, m.num_features:2*m.num_features]


    def forward(self, content, style):
        adain_params = self.mlp(style)
        self._assign_adain_params(adain_params)
        out = self.res(content)
        out = self.upsample(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.nets = nn.ModuleList()
        for _ in range(3):
            self.nets.append(self._make_net(num_channels))

    def _make_net(self, num_channels):
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

        return nn.Sequential(*model)

    def forward(self, x):
        outputs = []
        for model in self.nets:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs