import torch
import torch.nn as nn
import torch.nn.functional as F
import ConvBlockAttModule as cbam
import trblock as trb

class TrBlockSeries(nn.Module):
    def __init__(self, input_channel1,inout_channels,out_channels_last, num_blocks, kernel_size, stride=1, padding=1):
        super(TrBlockSeries, self).__init__()
        
        blocks = []
        current_channels = input_channels
        
        # Dynamically generate the specified number of conv layers
        for i in range(num_blocks):
            cbam = cbam.CBAM(in_channels = br_cha1,
                             reduction_ratio=8,
                             kernel_size=3)
            trblock  = trb.DepthwiseSeparableConv(br_cha1,
                                                    br_cha1,
                                                    kernel_size=3)
            
            blocks.append(cbam)
            blocks.append(trblock)
            
            layers.append(conv_layer)
            layers.append(nn.ReLU(inplace=True))  # Add ReLU activation
            
            # Update the current number of input channels
            current_channels = num_filters
        
        # Create the sequential model from the list of layers
        self.conv_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x

# Example usage
if __name__ == "__main__":
    # Create a ConvNet with 3 conv layers, input channels = 1 (grayscale), and 16 filters per conv layer
    num_layers = 3
    input_channels = 1
    num_filters = 16
    model = ConvNet(input_channels, num_layers, num_filters)
    
    # Print the model architecture
    print(model)
    
    # Example input (batch size = 1, 1 channel, 28x28 image)
    input_tensor = torch.randn(1, input_channels, 28, 28)
    output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
