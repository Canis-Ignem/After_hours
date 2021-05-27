
from torch.nn.modules.loss import CrossEntropyLoss
from model import Helon
import data_handler as dh
import torch
import numpy as np
from torch.autograd import Variable

x_train, y_train = dh.batchify("./datasets/train_data",5,512)

print(x_train.shape)

criterion = CrossEntropyLoss()

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
x, y = Variable(input), Variable(target)

if torch.cuda.is_available():
    
    x = input.cuda()
    y = target.cuda()

output = criterion(input, target)

print(output)

