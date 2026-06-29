import torch
import torch.nn as nn
import torch.nn.functional as F

class decoderModule(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1, padding1=1, padding2=1):
        super(decoderModule,self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        
        # Output layer
        self.output = nn.Conv2d(in_channels=out_channels2, out_channels=1, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        
        
    def forward(self, x):
        # Apply first convolutional layer followed by a ReLU activation
        x = F.relu(self.conv1(x))
        
        # Apply second convolutional layer followed by a ReLU activation
        x = F.relu(self.conv2(x))
        
        # Generate segmentation mask
        #x = F.relu(self.output(x))
        
        return x