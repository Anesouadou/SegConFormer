import torch
import torch.nn as nn
import torch.nn.functional as F

class CatModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CatModule,self).__init__()  
        self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x_seq):
        x = torch.cat(x_seq, dim=1)
        x = self.channel_reduction(x)
        return x

class AddModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AddModule,self).__init__()
        self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn                = nn.BatchNorm2d(out_channels)
        
    def forward(self,x_seq):
        x = torch.stack(x_seq)
        x = torch.sum(x,dim=0)
        x = self.channel_reduction(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    
class ProdModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProdModule,self).__init__()
        self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn                = nn.BatchNorm2d(out_channels)
        
    def forward(self,x_seq):
        x = torch.stack(x_seq)
        x = torch.prod(x,dim=0)
        x = self.channel_reduction(x)
        x = self.bn(x)
        x = F.relu(x)
        return x