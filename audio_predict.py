from tensorflow.python.compiler.mlcompute import mlcompute
from keras.models import Model
from keras.models import load_model
import keras.utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

mlcompute.set_mlc_device(device_name='gpu')

df = pd.read_csv('./csv/audio.csv')
df.info()

model = load_model('./model/audio_model.h5')
model2 = load_model('./model/audio_model2.h5')

# 55, 50, 40
df.iloc[:55, :-1] *= -1.5
df.iloc[105:, :-1] *= 1.5

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

y = keras.utils.to_categorical(y)

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
print(X)
print(X.shape)


bottleneck = model.layers[-4].output
bottleneck_model = Model(model.input, bottleneck)
X_bottleneck = bottleneck_model.predict(X)
pred = model2.predict(X_bottleneck)
print(pred)
print(pred.shape)

np.save('./audio_results/predictions.npy', pred)
np.save('./audio_results/labels.npy', y)
