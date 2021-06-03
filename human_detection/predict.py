
import PIL
import torch
import numpy as np
from PIL import Image


model = torch.load("./models/Best_model.pth")

def predict(frame):

    for i in range(frame.shape[0]):

        frame[i] = frame[i].resize((256,256))
    frame[i] = frame.reshape(24,3,256,256)
    img = torch.from_numpy(frame)
    
    with torch.no_grad():
        output = model(img.cuda())
        predictions = []
        for out in output.shape[0]:
            softmax = torch.exp(out).cpu()
            prob = list(softmax.numpy())
            predictions.append(np.argmax(prob, axis=1))
        predictions = np.array(predictions)
        print(np.bincount(predictions).argmax())