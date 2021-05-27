from PIL import Image
import os
import numpy as np
import torch

rng = np.random.default_rng(0)
'''
for image in os.listdir("./datasets"):
    img = Image.open("./datasets/"+image)
    print(np.array(img).shape)
'''

def train_test_split(pth, test_size):

    os.system("mkdir {}/train_data".format(pth))
    os.system("mkdir {}/test_data".format(pth))

    for image in os.listdir(pth):
        if os.path.isfile(pth+"/"+image):
            print(image)
            
            folder = rng.integers(low=0, high=100) > test_size*100

            if folder:
                os.system("mv {}/{} {}/train_data/".format(pth,image,pth))
            else:
                os.system("mv {}/{} {}/test_data/".format(pth,image,pth))

#print(len(os.listdir("datasets/test_data")))
#print(len(os.listdir("datasets/train_data")))


def batchify(pth, batch_size, img_size):
    
    x = []
    y = []

    for image in os.listdir(pth):
        
        #preprocessing the X
        img = Image.open(pth+"/"+image)
        img = img.resize((img_size,img_size)).convert("RGB")
        img = np.array(img, dtype=np.float32)

        #extracting the target
        target = 1 if image[0] == '1' or image[0] == 'p' else 0

        x.append(img)
        y.append(target)
        
    
    x = np.array(x)
    y = np.array(y)
    #print(x.shape)
    x = x.reshape(x.shape[0]//batch_size,batch_size,3,img_size,img_size)

    y = y.reshape(y.shape[0]//batch_size,batch_size)

    y = y.astype(int)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y

'''
img = Image.open("datasets/test_data/pperson_212.bmp")

print(np.array(img))

x , y = batchify("./datasets/test_data",5, 256)

img = Image.fromarray(x[0][0])

img.show()

print(x.shape)
print(y.shape)
'''
