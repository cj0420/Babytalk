import os
from tensorflow.python.compiler.mlcompute import mlcompute
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2
import pickle
import numpy as np
from collections import deque

mlcompute.set_mlc_device(device_name='gpu')

model = load_model('/content/drive/MyDrive/video_model.h5')
lb = pickle.loads(open('/content/drive/MyDrive/video_lb.pickle', "rb").read())

size = 128
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")


path = './video'

for dir in ['1', '2', '3']:
    folder = os.path.join(path, dir)
    files = os.listdir(folder)
    for vfile in files:
        src = os.path.join(folder, vfile)
        Q = deque(maxlen=size)
        cap = cv2.VideoCapture(src)
        print(vfile)
        while (cap.isOpened()):
            # read the next frame from the file
            grabbed, frame = cap.read()

            # if the frame was not grabbed, then we have reached the end of the stream
            if grabbed == False:
                break

            # clone the output frame, then convert it from BGR to RGB
            # ordering, resize the frame to a fixed 224x224, and then
            # perform mean subtraction
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)).astype("float32")
            frame -= mean
            frame = np.expand_dims(frame, axis=0)

            # make predictions on the frame and then update the predictions
            # queue
            preds = model.predict(frame)
            Q.append(preds)
            #print('predictions: ', preds)

            # perform prediction averaging over the current history of
            # previous predictions
            results = np.array(Q).mean(axis=0)
            #print('results:', results)
            i = np.argmax(results)
            label = lb.classes_[i]

        print('results:', results)
        np.save(os.path.join('/content/npy', dir,
                vfile.replace('mp4', 'npy')), results)
