## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer
        
        # **CONV2D-1**
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W - F)/S + 1 = (224 - 5)/1 + 1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after Maxpool layer, this becomes (32, 110, 110)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # **Maxpool layer**
        # pool with kernel_size=2, stride=2
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # **CONV2D-2**
        # 32 input image channel (grayscale), 64 output channels/feature maps, 4x4 square convolution kernel
        ## output size = (W - F)/S + 1 = (110 - 4)/1 + 1 = 107
        # the output Tensor for one image, will have the dimensions: (64, 107, 107)
        # after Maxpool layer, this becomes (64, 53, 53)
        
        self.conv2 = nn.Conv2d(32, 64, 4)
        
        # **CONV2D-3**
        # 64 input image channel (grayscale), 128 output channels/feature maps, 3x3 square convolution kernel
        ## output size = (W - F)/S + 1 = (53 - 3)/1 + 1 = 51
        # the output Tensor for one image, will have the dimensions: (128, 51, 51)
        # after Maxpool layer, this becomes (128, 25, 25)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # **CONV2D-4**
        # 1 input image channel (grayscale), 32 output channels/feature maps, 2x2 square convolution kernel
        ## output size = (W - F)/S + 1 = (25 - 3)/1 + 1 = 23
        # the output Tensor for one image, will have the dimensions: (256, 23, 23)
        # after Maxpool layer, this becomes (256, 11, 11)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # Fully Connected Layers
        # 512 outputs * the 12*12 feature maps size
        self.fc1 = nn.Linear(256*11*11, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 68*2)
        
        # Dropout Layers
        self.drop1 = nn.Dropout(p = 0.2)
        self.drop2 = nn.Dropout(p = 0.2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to             avoid overfitting
        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # 5 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropouts in between
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        
        
        return x
