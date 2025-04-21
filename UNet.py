import torch # importing pytorch library
import torch.nn as nn # importing pytorch's neural network module

'''
The DoubleConvolutionNN class is a building block of the UNet architecture.
It consists of two convolutional layers, each followed by batch normalization and the ReLU activation function.
The first convolutional layer takes input_channels as input and outputs output_channels.
The second convolutional layer takes output_channels as input and outputs output_channels.
The kernel size is 3x3 and padding is 1 to maintain the spatial dimensions of the input.
'''
class DoubleConvolutionNN(nn.Module):
    """
    DoubleConvolutionNN inherits from nn.Module
    """
    def __init__(self, input_channels, output_channels):
        super().__init__() # calling the constructor of the parent class nn.Module which initializes the module

        self.doubleconv = nn.Sequential( # nn.Sequential is a allows us to define a sequence of layers
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, bias = False), # nn.Conv2d applies sliding kernels to a 2D input. The kernel size is 3x3 and padding is 1 to maintain the spatial dimensions of the input. Output channels is the number of filters we want to apply to the input image.
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input_image):
        return self.doubleconv(input_image)


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, features_list = [64, 128, 256, 512]):
        super().__init__()
        
        self.downward = nn.ModuleList()
        self.upwards = nn.ModuleList()
        self.half_size = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of the UNet
        for feature in features_list:
            step = DoubleConvolutionNN(input_channels, feature)
            self.downward.append(step)
            input_channels = feature

        # Middle part (bottleneck) of the UNet
        self.bottleneck = DoubleConvolutionNN(features_list[-1], 2*features_list[-1])

        # Up part of the UNet
        for feature in features_list[::-1]:
            double_size = nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            self.upwards.append(double_size)
            step = DoubleConvolutionNN(feature * 2, feature)
            self.upwards.append(step)
        
        self.finalstep = DoubleConvolutionNN(features_list[0], output_channels)
    
    def forward(self, input_image):
        downward_step_connection = []

        for downward_convolution in self.downward:
            input_image = downward_convolution(input_image)
            downward_step_connection.append(input_image)
            input_image = self.half_size(input_image)

        input_image = self.bottleneck(input_image)
        downward_step_connection.reverse()

        for upward_convolution_index in range(0, len(self.upwards), 2):
            input_image = self.upwards[upward_convolution_index](input_image)
            connection = downward_step_connection[upward_convolution_index//2]

            if connection.shape != input_image.shape:
                input_image = nn.functional.interpolate(input_image, connection.shape[2:], mode = "bilinear")

            input_image = torch.cat((input_image, connection), dim = 1)
            input_image = self.upwards[upward_convolution_index + 1](input_image)

        return self.finalstep(input_image)
