import tensorflow as tf
from tensorflow.python.compiler.mlcompute import mlcompute
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Concatenate
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

mlcompute.set_mlc_device(device_name='gpu')

audio_path = "./audio_results"
video_path = "./video_results"
audio_results = np.load("./audio_results/predictions.npy")
video_results = []
labels = np.load("./audio_results/labels.npy")

for dir in ['hugging', 'hungry', 'sleepy']:
    folder = os.path.join(video_path, dir)
    files = os.listdir(folder)
    for file in files:
        src = os.path.join(folder, file)
        npy = np.load(src)[0]
        video_results.append(npy)

video_results = np.array(video_results)

X1_train, X1_test, y_train, y_test = train_test_split(
    video_results, labels, test_size=0.2, random_state=10)
X2_train, X2_test, y_train, y_test = train_test_split(
    audio_results, labels, test_size=0.2, random_state=10)


audio_output = Input(shape=(3,))
video_output = Input(shape=(3,))
concat = Concatenate(axis=1)([audio_output, video_output])
concat = Flatten()(concat)
concat = Dense(32, activation='relu')(concat)
concat_output = Dense(3, activation='softmax')(concat)

concatModel = Model(inputs=[audio_output, video_output], outputs=concat_output)
print(concatModel.summary())

opt_rms = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-4)
concatModel.compile(loss='binary_crossentropy',
                    optimizer=opt_rms, metrics=['accuracy'])

history = concatModel.fit([X1_train, X2_train], y_train,
                          epochs=100,  validation_data=([X1_test, X2_test], y_test))

plt.style.use('ggplot')
plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./output/total_acc.jpg')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./output/total_loss.jpg')
plt.show()
