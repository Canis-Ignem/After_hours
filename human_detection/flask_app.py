#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import numpy as np
from PIL import Image
import torch
from torchsummary import summary
from predictor import predict



app = Flask(__name__)

cap = cv2.VideoCapture(0)

frames = []


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera, frames):
    """Video streaming generator function."""
    i = 0
    while True:
        success, frame = camera.read()
        if i < 24:
            pil_im = Image.fromarray(frame)
            frames.append(pil_im)
            i += 1
        else:
            predict(frames)
            frames = []
            i = 0
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(cap,[]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)