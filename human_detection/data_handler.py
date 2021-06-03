import cv2
from PIL import Image
import os
import numpy as np
import torch
rng = np.random.default_rng(0)

def train_test_split(dir, test_size):

    os.system("mkdir ./{}/test_dir".format(dir))
    os.system("mkdir ./{}/train_dir".format(dir))
    


    
    if os.path.isdir(dir):
        
        for file in os.listdir(dir):
            if os.path.isfile(dir+file):
                print(file)
                folder = rng.integers(low=0, high= 100) > (test_size*100)
                
                if folder:
                    os.system("mv {}/{} ./{}/train_dir/".format(dir,file,dir))
                else:
                    os.system("mv {}/{} ./{}/test_dir/".format(dir,file,dir))
    

#train_test_split("datasets/",0.1)      

def create_batch(pth, batch_size,im_size):
    
    x = []
    y = []
    if os.path.isdir(pth):
        
        for file in os.listdir(pth):
            img = Image.open(pth+file)
            img = img.resize((im_size,im_size)).convert('RGB')
            #t = np.zeros(2, dtype=int)
            target = 1 if file[0] == '1' or file[0] == 'p' else 0
            #t[target] = 1
            img = np.array(img,dtype=np.float32)
            x.append(img)
            y.append(target)
    x = np.array(x)
    y = np.array(y)

    x = x.reshape(x.shape[0]//batch_size, batch_size,3, x.shape[2],x.shape[2])
    x = torch.from_numpy(x)

    y = y.astype(int);
    y = y.reshape(y.shape[0]//batch_size,batch_size)
    y = torch.from_numpy(y)


    return x,y

