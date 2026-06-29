import torch
import torch.nn as nn
from packages import fusion

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depth-wise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        
        # Point-wise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        # Apply depth-wise convolution
        x = self.depthwise(x)
        # Apply point-wise convolution
        x = self.pointwise(x)
        return x


class ThreePhasetrblock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size1, padding1,kernel_size2, padding2,kernel_size3, padding3,stride=1):
        super(ThreePhasetrblock, self).__init__()
        # phase1 3x3
        self.phase1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size1, stride=stride, padding=padding1, groups=in_channels, bias=False)        
        
        # phase2 5x5
        self.phase2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size2, stride=stride, padding=padding2, groups=in_channels, bias=False)

        # phase3 7x7
        self.phase3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size3, stride=stride, padding=padding3, groups=in_channels, bias=False)

        # phase addition
        self.addphases = fusion.AddModule(in_channels=out_channels, out_channels=out_channels) 

        # pointwise conv
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self,x):
        # Generate the three phases
        x_phase1 = self.phase1(x)
        x_phase2 = self.phase2(x)
        x_phase3 = self.phase3(x)

        # Add the three phases 
        phases_srq = (x_phase1,x_phase2,x_phase3)
        added_phases = self.addphases(phases_srq)

        # Apply 
        output = x*added_phases 
        # multiply the added three phases with input 
        output = self.pointwise(output)
        return output

        

    