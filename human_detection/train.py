from numpy.core.numeric import outer
from model import CNN
import data_handler as dh
import torch
from torch.autograd  import Variable
from torch.nn import L1Loss, CrossEntropyLoss, BCELoss
from torch.optim import Adam, SGD
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_x,train_y = dh.create_batch("./datasets/train_dir/",5,512)
print(train_x.shape)

val_x,val_y = dh.create_batch("./datasets/test_dir/",5,512)
print(val_x.shape)


train_losses = []
val_losses = []



def train(epochs,train_x,train_y):
    best_val = 2
    model.train()
    for epoch in range(epochs):
        
        rnd = np.random.permutation(train_x.shape[0]) 
        train_x = train_x[rnd]
        train_y = train_y[rnd]

        tr_loss = 0
        
        for x , y in zip(train_x,train_y):
    
            x, y = Variable(x), Variable(y)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()


            optimizer.zero_grad()
            
            output_train = model(x)

            #output_train = torch.reshape(output_train,(output_train.shape[0],))
            #print(y)
            #print(output_train)

            loss_train = criterion(output_train, y)
            #print(loss_train)

            loss_train.backward()

            optimizer.step()

            tr_loss += loss_train.item()
        '''
        if epoch%2 == 0:
            train_losses.append(tr_loss/train_x.shape[0])
            print('Epoch : ',epoch, "\t Train loss: ", tr_loss/train_x.shape[0])
        '''
        if epoch%1 == 0:
            train_losses.append(tr_loss/train_x.shape[0])
            
            val_loss = 0
            with torch.no_grad():
                for x,y in zip(val_x,val_y):
                    output = model(x.cuda())
                    #output = torch.reshape(output,(output.shape[0],))
                    #print(output.shape)
                    loss_val = criterion(output, y.cuda())
                    val_loss += loss_val.item()

            val_loss =  val_loss/val_x.shape[0]
            val_losses.append(val_loss)
            print('Epoch : ',epoch, "\t Train loss: ", tr_loss/train_x.shape[0], "\t Validation loss: ", val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model, "./models/Best_model2.pth")
        scheduler.step()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()

def test(model, x_data, y_data):

    predictions = []
    # prediction for training set
    with torch.no_grad():
        for x in x_data:

            output = model(x.cuda()).cpu()
            #print(output.shape)
            preds = np.argmax(output, axis=1)
            predictions.append(preds.numpy())
            #print(preds)
            
    predictions = np.array(predictions).flatten()
    y_data = y_data.numpy().flatten()
    #print(predictions)
    #print(y_data)

    print(accuracy_score(y_data, predictions))



model = CNN()

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

# defining the loss function
criterion = CrossEntropyLoss()


# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

summary(model,(3, 512, 512))

best_val = 2

# defining the number of epochs
epochs = 5
#train(epochs,train_x,train_y)


model = torch.load("./models/Best_model.pth")


print("Trainning accuracy is:")
test(model, train_x, train_y)
print("Testing accuracy is:")
test(model, val_x, val_y)





