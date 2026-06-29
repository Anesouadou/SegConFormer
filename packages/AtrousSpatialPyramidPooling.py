import torch
import torch.nn as nn
import torch.nn.functional as F

# Four branches
class asppblock(nn.Module):
    def __init__(self, branch_options):
        super(asppblock, self).__init__()
        # Branch1
        branch1       = branch_options['branch1']
        in_channels1  = branch1['in_channels']
        out_channels1 = branch1['out_channels']
        kernel_size1  = branch1['kernel_size']
        dilation1     = branch1['dilation']
        stride1       = branch1['stride']
        padding1      = branch1['padding']
        
        self.asppc1  = nn.Conv2d(in_channels1,
                                 out_channels1,
                                 kernel_size=kernel_size1,
                                 padding=padding1,
                                 stride=stride1,
                                 dilation=dilation1) 
        self.asppb1  = nn.BatchNorm2d(out_channels1)
        self.asppa1  = nn.ReLU()
        
        # Branch2
        branch2       = branch_options['branch2']
        in_channels2  = branch2['in_channels']
        out_channels2 = branch2['out_channels']
        kernel_size2  = branch2['kernel_size']
        dilation2     = branch2['dilation']
        stride2       = branch2['stride']
        padding2      = branch2['padding']
        
        self.asppc2  = nn.Conv2d(in_channels2,
                                 out_channels2,
                                 kernel_size=kernel_size2,
                                 padding=padding2,
                                 stride=stride2,
                                 dilation=dilation2) 
        self.asppb2  = nn.BatchNorm2d(out_channels2)
        self.asppa2  = nn.ReLU()
        
        # Branch3
        branch3       = branch_options['branch3']
        in_channels3  = branch3['in_channels']
        out_channels3 = branch3['out_channels']
        kernel_size3  = branch3['kernel_size']
        dilation3     = branch3['dilation']
        stride3       = branch3['stride']
        padding3      = branch3['padding']
        
        self.asppc3  = nn.Conv2d(in_channels3,
                                 out_channels3,
                                 kernel_size=kernel_size3,
                                 padding=padding3,
                                 stride=stride3,
                                 dilation=dilation3) 
        self.asppb3  = nn.BatchNorm2d(out_channels3)
        self.asppa3  = nn.ReLU()
        
        # Branch4
        branch4       = branch_options['branch4']
        in_channels4  = branch4['in_channels']
        out_channels4 = branch4['out_channels']
        kernel_size4  = branch4['kernel_size']
        dilation4     = branch4['dilation']
        stride4       = branch4['stride']
        padding4      = branch4['padding']
        
        self.asppc4  = nn.Conv2d(in_channels4,
                                 out_channels4,
                                 kernel_size=kernel_size4,
                                 padding=padding4,
                                 stride=stride4,
                                 dilation=dilation4) 
        self.asppb4  = nn.BatchNorm2d(out_channels4)
        self.asppa4  = nn.ReLU()
        

        
    def forward(self, x):
        # Apply first convolutional layer followed by a ReLU activation
        xb1 = self.asppa1(self.asppb1(self.asppc1(x)))
        xb2 = self.asppa2(self.asppb2(self.asppc2(x)))
        xb3 = self.asppa3(self.asppb3(self.asppc3(x)))
        xb4 = self.asppa4(self.asppb4(self.asppc4(x)))
        

        
        return xb1,xb2,xb3,xb4

# Five branches
class asppblock5(nn.Module):
    def __init__(self, branch_options):
        super(asppblock5, self).__init__()
        # Branch1
        branch1       = branch_options['branch1']
        in_channels1  = branch1['in_channels']
        out_channels1 = branch1['out_channels']
        kernel_size1  = branch1['kernel_size']
        dilation1     = branch1['dilation']
        stride1       = branch1['stride']
        padding1      = branch1['padding']
        
        self.asppc1  = nn.Conv2d(in_channels1,
                                 out_channels1,
                                 kernel_size=kernel_size1,
                                 padding=padding1,
                                 stride=stride1,
                                 dilation=dilation1) 
        self.asppb1  = nn.BatchNorm2d(out_channels1)
        self.asppa1  = nn.ReLU()
        
        # Branch2
        branch2       = branch_options['branch2']
        in_channels2  = branch2['in_channels']
        out_channels2 = branch2['out_channels']
        kernel_size2  = branch2['kernel_size']
        dilation2     = branch2['dilation']
        stride2       = branch2['stride']
        padding2      = branch2['padding']
        
        self.asppc2  = nn.Conv2d(in_channels2,
                                 out_channels2,
                                 kernel_size=kernel_size2,
                                 padding=padding2,
                                 stride=stride2,
                                 dilation=dilation2) 
        self.asppb2  = nn.BatchNorm2d(out_channels2)
        self.asppa2  = nn.ReLU()
        
        # Branch3
        branch3       = branch_options['branch3']
        in_channels3  = branch3['in_channels']
        out_channels3 = branch3['out_channels']
        kernel_size3  = branch3['kernel_size']
        dilation3     = branch3['dilation']
        stride3       = branch3['stride']
        padding3      = branch3['padding']
        
        self.asppc3  = nn.Conv2d(in_channels3,
                                 out_channels3,
                                 kernel_size=kernel_size3,
                                 padding=padding3,
                                 stride=stride3,
                                 dilation=dilation3) 
        self.asppb3  = nn.BatchNorm2d(out_channels3)
        self.asppa3  = nn.ReLU()
        
        # Branch4
        branch4       = branch_options['branch4']
        in_channels4  = branch4['in_channels']
        out_channels4 = branch4['out_channels']
        kernel_size4  = branch4['kernel_size']
        dilation4     = branch4['dilation']
        stride4       = branch4['stride']
        padding4      = branch4['padding']
        
        self.asppc4  = nn.Conv2d(in_channels4,
                                 out_channels4,
                                 kernel_size=kernel_size4,
                                 padding=padding4,
                                 stride=stride4,
                                 dilation=dilation4) 
        self.asppb4  = nn.BatchNorm2d(out_channels4)
        self.asppa4  = nn.ReLU()

        # Branch5
        branch5       = branch_options['branch5']
        in_channels5  = branch5['in_channels']
        out_channels5 = branch5['out_channels']
        kernel_size5  = branch5['kernel_size']
        dilation5     = branch5['dilation']
        stride5       = branch5['stride']
        padding5      = branch5['padding']
        
        self.asppc5  = nn.Conv2d(in_channels5,
                                 out_channels5,
                                 kernel_size=kernel_size5,
                                 padding=padding5,
                                 stride=stride5,
                                 dilation=dilation5) 
        self.asppb5  = nn.BatchNorm2d(out_channels5)
        self.asppa5  = nn.ReLU()
        

        
    def forward(self, x):
        # Apply first convolutional layer followed by a ReLU activation
        xb1 = self.asppa1(self.asppb1(self.asppc1(x)))
        xb2 = self.asppa2(self.asppb2(self.asppc2(x)))
        xb3 = self.asppa3(self.asppb3(self.asppc3(x)))
        xb4 = self.asppa4(self.asppb4(self.asppc4(x)))
        xb5 = self.asppa5(self.asppb5(self.asppc5(x)))
        

        
        return xb1,xb2,xb3,xb4,xb5
