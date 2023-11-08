import torch
import torch.nn as nn

class AutoLinear(nn.Module):
    def __init__(self, out_features, device, in_features=0):
        super(AutoLinear, self).__init__()
        self.device = device
        self.out_features = out_features
        self.linear = None
        if in_features != 0:
            self.linear = nn.Linear(in_features, self.out_features).to(self.device)

    def forward(self, x):
        if self.linear == None:
            in_features = x.size(1)
            self.linear = nn.Linear(in_features, self.out_features).to(self.device)
        return self.linear(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, max_pool):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if max_pool==1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Model(nn.Module):
    def __init__(self, num_classes, device):
        super(Model, self).__init__()
        #CNN (N, C, H, W)

        self.conv_blocks = nn.ModuleList()
        
        #in_channels, out_channels, kernel_size, stride, max_pool(1 or 0)
        conv_params = [[1,64,5,2,1], [64,64,3,1,1], [64,64,3,1,0], [64,64,3,1,0], [64,64,3,1,0], [64,64,3,1,0],[64,64,3,1,0], [64,128,3,1,0]]

        for param in conv_params:
            conv_block = ConvBlock(*param)
            self.conv_blocks.append(conv_block)

        hidden_features = 1024
        self.fc1 = AutoLinear(hidden_features, in_features=86016, device=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # Define the forward pass with BatchNorm
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        if self.training:
            return x
        else:
            return x.argmax(dim=1).to('cpu')