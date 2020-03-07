import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        self.dp_value = 0.1
        super(Net, self).__init__()
        # Input Block
        self.convblock1_DL = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=0, bias=False, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dp_value),
        ) # output_size = 30, RF = 5

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dp_value),
        ) # output_size = 28, RF = 7

        # CONVOLUTION BLOCK 1
        self.SeperableConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dp_value),
        ) # output_size = 26, RF = 9

        # CONVOLUTION BLOCK 1
        self.DepthConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, groups=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dp_value),
        ) # output_size = 24, RF = 11




        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 11
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dp_value),
        ) # output_size = 12, RF = 15
        

        # CONVOLUTION BLOCK 2
        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=0, bias=False),
            #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dp_value),
        ) # output_size = 15, RF = 19

        # CONVOLUTION BLOCK 2
        self.DepthConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dp_value),
        ) # output_size = 13, RF = 23

        # CONVOLUTION BLOCK 2
        self.DepthConvBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, groups=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dp_value),
        ) # output_size = 11, RF = 27

        self.SeperableConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,1), padding=1, groups=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,3)),
        ) # output_size = 9, RF = 31
      
        self.DepthConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1, groups=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1)),
        ) # output_size = 9, RF = 35
        # OUTPUT BLOCK
        self.GAP = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        ) # output_size = 1

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.convblock1_DL(x)
        x = self.convblock2(x)
        x = self.SeperableConvBlock1(x)
        x = self.DepthConvBlock1(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.ConvBlock4(x)
        x = self.DepthConvBlock2(x)
        x = self.DepthConvBlock3(x)
        x = self.SeperableConvBlock2(x)
        #x = self.DepthConvBlock4(x)

        x = self.GAP(x)        
        x = self.convblock4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)