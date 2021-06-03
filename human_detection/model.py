import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.nn.modules import batchnorm
from torch.optim import Adam, SGD
import data_handler as dh
from torch import optim, softmax
from tqdm import tqdm
import numpy as np
from torchsummary import summary


class CNN(Module):
    
    def __init__(self):
        super(CNN,self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 512, kernel_size=3, stride=2),
            Dropout(0.2, inplace= True),
            BatchNorm2d(512),
            ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=2),
            Conv2d(512, 256, kernel_size=3, stride=2),
            Dropout(0.2, inplace= True),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 128, kernel_size=3, stride=2),
            Dropout(0.2, inplace= True),
            BatchNorm2d(128),
            ReLU(),
            #MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = Sequential(
            Linear(128*63*63,512),
            Dropout(0.2, inplace= True),
            Linear(512,256),
            Dropout(0.2, inplace= True),
            Linear(256,128),
            Dropout(0.2, inplace= True),
            Linear(128,64),
            Dropout(0.2, inplace= True),
            Linear(64,2),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
