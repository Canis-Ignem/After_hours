#model
from numpy.random import gamma
from torch import device
import torch
from model import Helon

#Data
import data_handler as dh
import numpy as np
from matplotlib import pyplot as plt

#torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, L1Loss, BCELoss, Softmax
from torch.optim import Adam
from torchsummary import summary

#metrics
from sklearn.metrics import accuracy_score


x_train, y_train = dh.batchify("./datasets/train_data",5,512)
print(x_train.shape)

x_val, y_val = dh.batchify("./datasets/test_data",5,512)
print(x_val.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loss = []
val_loss = []

def train(x_train, y_train, x_val, y_val, model, epochs, optimizer, criterion, scheduler):

    best_val = 2
    model.train()

    for epoch in range(epochs+1):

        rnd = np.random.permutation(x_train.shape[0])
        x_train = x_train[rnd]
        y_train = y_train[rnd]
        
        e_loss = 0

        for x, y in zip(x_train, y_train):

            x, y = Variable(x), Variable(y)

            if torch.cuda.is_available():

                x = x.cuda()
                y = y.cuda()
            
            optimizer.zero_grad()

            output = model(x)

            tr_loss = criterion(output, y)

            tr_loss.backward()
            optimizer.step()

            e_loss += tr_loss.item() 

        train_loss.append(e_loss/x_train.shape[0])

        #Logs and validation
        if epoch % 2 == 0:
        
            validation_loss = 0

            with torch.no_grad():

                for x, y in zip(x_val, y_val):
    
                    x, y = Variable(x), Variable(y)

                    if torch.cuda.is_available():

                        x = x.cuda()
                        y = y.cuda()
                    
                    val_output = model(x)

                    v_loss = criterion(val_output, y)

                    validation_loss += v_loss.item()

                avg_val_loss = validation_loss/x_val.shape[0]
                val_loss.append(avg_val_loss)
                print("Epoch: ", epoch, " \t train_loss: ", train_loss[-1], "\t validation loss: ", avg_val_loss)

                if best_val > avg_val_loss:

                    best_val = avg_val_loss
                    torch.save(model,"./models/Best_Model2.pth")
        
        scheduler.step()
    print(len(train_loss))
    print(len(val_loss))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.show()



helon = Helon()
criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    helon.cuda()
    criterion.cuda()

summary(helon,(3, 512, 512))

optim = Adam(helon.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma= 0.95)



epochs = 0

train(x_train, y_train, x_val, y_val, helon, epochs, optim, criterion, scheduler)
