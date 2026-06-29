import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2,method='transpose', scale_factor=2,padding=0,output_padding=1):
        super(UpsampleModule, self).__init__()
        self.method = method
        self.scale_factor = scale_factor
        
        if method == 'transpose':
            # Transpose convolution (also known as deconvolution)
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=scale_factor, 
                padding=padding, 
                output_padding=output_padding
            )
        elif method == 'bilinear':
            # Bilinear interpolation does not require a learnable layer
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        elif method == 'nearest':
            # Nearest neighbor interpolation does not require a learnable layer
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        else:
            raise ValueError("Unsupported upsampling method: {}".format(method))

    def forward(self, x):
        if self.method == 'transpose':
            x = self.upsample(x)
        elif self.method == 'bilinear':
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            x = self.upsample(x)
        elif self.method == 'nearest':
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
            x = self.upsample(x)
        return x
