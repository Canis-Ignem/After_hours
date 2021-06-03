import torch
from torchsummary import summary
import numpy as np
from IPython.display import Audio
from audio import alarm

model = torch.load("./models/Best_model.pth")
summary(model,(3, 512, 512))


sound_file = "alarms/jon.wav"

def predict(frames):
    
    for i in range(len(frames)):
        #img = Image.fromarray(frames[i])
        frames[i] =  frames[i].resize((512,512))
        frames[i] = np.array(frames[i], dtype = np.float32)
    frames = np.array(frames)
    frames = frames.reshape(24,3,512,512)
    img = torch.from_numpy(frames)
    
    with torch.no_grad():
        output = model(img.cuda())
        predictions = []
        for out in range(output.shape[0]):
            softmax = torch.exp(output[out]).cpu()
            prob = list(softmax.numpy())
            predictions.append(np.argmax(prob, axis=0))
        predictions = np.array(predictions)
        
        final_pred = np.bincount(predictions).argmax()
        print(final_pred)
     
        if final_pred == 1:
            print("Persona")
            #alarm(sound_file)