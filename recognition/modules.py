import torch
import torch.nn as nn
from torchvision.transforms import v2




class contextBlock(nn.Module):
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

    


class localisationBlock(nn.Module):
    """
    Localisation module
    Contains one 3x3x3 convolution, then a 1x1x1 convolution
    """
    pass


class upsampleBlock(nn.Module):
    """
    Upsampling module
    Repeats feature voxels twice in each dimension
    Performs one 3x3x3 convolution
    """
    pass


class UNet2D(nn.Module):
    """
    Class implementing blocks, adding residuals and segmentation layers
    in upsampling half of the network
    """
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
    pass