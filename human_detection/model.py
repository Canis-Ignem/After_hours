import torch
from torch.nn import Conv2d,  MaxPool2d, ReLU, Sequential, BatchNorm2d, Dropout, Module
from torch.nn.modules.linear import Linear



class Helon(Module):
    def __init__(self):
        super(Helon,self).__init__()

        self.cnn_layers = Sequential(

            Conv2d(3,512,kernel_size= 3, stride = 2),
            Dropout(0.4, inplace=True),
            BatchNorm2d(512),
            ReLU(inplace=True),

            Conv2d(512,256,kernel_size= 3, stride = 2),
            Dropout(0.4, inplace=True),
            BatchNorm2d(256),
            ReLU(inplace=True),

            Conv2d(256,128,kernel_size= 3, stride = 2),
            Dropout(0.4, inplace=True),
            BatchNorm2d(128),
            ReLU(),
        )

        self.linear_layers = Sequential(

            Linear(128 * 63 * 63, 512),
            Dropout(0.4, inplace=True),
            Linear(512, 256),
            Dropout(0.4, inplace=True),
            Linear(256, 128),
            Dropout(0.4, inplace=True),
            Linear(128, 64),
            Dropout(0.4, inplace=True),
            Linear(64, 2),

        )
    
    def forward(self, x):

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x