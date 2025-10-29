import torch
import torch.nn as nn

class ContextBlock(nn.Module):
    """
    Context block
    contains two 3x3x3 convolutions and a dropout layer
    """
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.act = nn.LeakyReLU(10e-2)

        self.context = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            self.act,
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
    
    def forward(self, x):
        identity = x
        out = self.context(x)
        out += identity
        out = self.act(out)
        return out


class LocalisationBlock(nn.Module):
    """
    Localisation module
    Contains one 3x3x3 convolution, then a 1x1x1 convolution which halves channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.act = nn.LeakyReLU(10e-2)

        self.localisation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            self.act,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.localisation(x)

class UpsampleBlock(nn.Module):
    """
    Upsampling module
    Repeats feature voxels twice in each dimension
    Performs one 3x3x3 convolution
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(10e-2)
        )
    
    def forward(self, x):
        return self.upsample(x)

class UNet2D(nn.Module):
    """
    Class implementing blocks, adding residuals and segmentation layers
    in upsampling half of the network
    """
    def __init__(self, in_channels, latent_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # ENCODING
        # each encoding layer doubles the amount of feature maps, so input of next layer is output of previous
        # and output of next layer is twice its input
        # perform downsampling at each layer with stride 2 convolution
        self.encode1 = ContextBlock(in_channels, latent_channels)
        self.ds1 = nn.Conv2d(latent_channels, latent_channels * 2, kernel_size=3, stride=2, padding=1)

        self.encode2 = ContextBlock(latent_channels * 2, latent_channels * 2)
        self.ds2 = nn.Conv2d(latent_channels * 2, latent_channels * 4, kernel_size=3, stride=2, padding=1)

        self.encode3 = ContextBlock(latent_channels * 4, latent_channels * 4)
        self.ds3 = nn.Conv2d(latent_channels * 4, latent_channels * 8, kernel_size=3, stride=2, padding=1)

        self.encode4 = ContextBlock(latent_channels * 8, latent_channels * 8)
        self.ds4 = nn.Conv2d(latent_channels * 8, latent_channels * 16, kernel_size=3, stride=2, padding=1)

        # bottleneck layer
        self.bottle = ContextBlock(latent_channels * 16, latent_channels * 16)

        # DECODING

        # each localisation block has concatenated output of corresponding encoding layer
        # therefore latent_channels will be double output of previous upsample block
        self.us1 = UpsampleBlock(latent_channels*16, latent_channels * 8)
        self.localise1 = LocalisationBlock(latent_channels * 16, latent_channels * 8)

        self.us2 = UpsampleBlock(latent_channels * 8, latent_channels * 4)
        self.localise2 = LocalisationBlock(latent_channels * 8, latent_channels * 4)

        # DEEP SUPERVISION
        # perform segmentation convolution at earlier points and sum
        # note that 2 and 3 corresponds to localise 2 and 3 as outputs of these used for
        # segmentation
        self.seg_local2 = nn.Conv2d(latent_channels * 4, num_classes, kernel_size=1)
        self.upsample_seg = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        self.us3 = UpsampleBlock(latent_channels * 4, latent_channels * 2)
        self.localise3 = LocalisationBlock(latent_channels * 4, latent_channels * 2)
        self.seg_local3 = nn.Conv2d(latent_channels * 2, num_classes, kernel_size=1)
        self.us4 = UpsampleBlock(latent_channels * 2, latent_channels)

        # SEGMENTATION
        self.final = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        self.segment = nn.Conv2d(latent_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # ENCODER
        # store each output for encoding for U skip connections
        # convolution layer between each contextblock
        e1 = self.encode1(x)
        e2 = self.encode2(self.ds1(e1))
        e3 = self.encode3(self.ds2(e2))
        e4 = self.encode4(self.ds3(e3))
        # BOTTLENECK
        x_middle = self.bottle(self.ds4(e4))
        
        # DECODER
        # concatenate corresponding encoder layers
        u1 = self.us1(x_middle)
        concat1 = torch.cat([u1, e4], dim=1)
        d1 = self.localise1(concat1)

        u2 = self.us2(d1)
        concat2 = torch.cat([u2, e3], dim=1)
        d2 = self.localise2(concat2)

        # first deep supervision segmentation
        seg1 = self.upsample_seg(self.seg_local2(d2))

        u3 = self.us3(d2)
        concat3 = torch.cat([u3, e2], dim=1)
        d3 = self.localise3(concat3)

        # second deep supervision segmentation
        seg2 = self.seg_local3(d3)
        # sum deep supervision layers and upsample to match final layer output
        seg_sum = self.upsample_seg(seg1 + seg2)
        u4 = self.us4(d3)
        concat4 = torch.cat([u4, e1], dim=1)

        final = self.final(concat4)
        seg_final = self.segment(final)
        return seg_final + seg_sum




        