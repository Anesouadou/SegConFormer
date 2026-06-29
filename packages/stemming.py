import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerConvNet(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1, padding1=1, padding2=1):
        super(TwoLayerConvNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.bn2 = nn.BatchNorm2d(out_channels2)
        
    def forward(self, x):
        # Apply first convolutional layer followed by a ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Apply second convolutional layer followed by a ReLU activation
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        return out

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1, padding1=1, padding2=1):
        super(ThreeLayerConvNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.bn2 = nn.BatchNorm2d(out_channels2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=out_channels2, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.bn3 = nn.BatchNorm2d(out_channels2)
        
    def forward(self, x):
        # Apply first convolutional layer followed by a ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Apply second convolutional layer followed by a ReLU activation
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Apply third convolutional layer followed by a ReLU activation
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        
        return out

class FourLayerConvNet(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1=3, kernel_size2=3, stride1=1, stride2=1, padding1=1, padding2=1):
        super(FourLayerConvNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.bn2 = nn.BatchNorm2d(out_channels2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=out_channels2, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.bn3 = nn.BatchNorm2d(out_channels2)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=out_channels2, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.bn4 = nn.BatchNorm2d(out_channels2)
        
    def forward(self, x):
        # Apply first convolutional layer followed by a ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Apply second convolutional layer followed by a ReLU activation
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Apply third convolutional layer followed by a ReLU activation
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Apply fourth convolutional layer followed by a ReLU activation
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        
        return out

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        # First convolutional layer of the ResNet block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer of the ResNet block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer if necessary to match dimensions
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
