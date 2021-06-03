import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.nn.modules import batchnorm
from torch.nn.modules.loss import BCELoss
from torch.optim import Adam, SGD
import data_handler as dh
from torch import optim, softmax
from tqdm import tqdm
import numpy as np

a = torch.from_numpy(np.array([0.3,0.44]))
s = ReLU()
print(s(a))


b = np.array([[0,1],[1,0]])

print( np.argmax(b, axis=1))