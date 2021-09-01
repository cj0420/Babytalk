from unicodedata import name
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import mlcompute
from tensorflow.keras import Model, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

mlcompute.set_mlc_device(device_name='gpu')

path = './frame'
# initialize the set of labels from the spots activity dataset we are
# going to train our network on
epochs = 10
plot_acc = './output/video_acc_plot.png'
plot_loss = './output/video_loss_plot.png'

data = []
labels = []

'''
for dir in ["hugging", "hungry", "sleepy"]:
    folder = os.path.join(path, dir)
    files = os.listdir(folder)

    for file in files:
        if file == '.DS_Store':
            continue
        src = os.path.join(folder, file)

        # load the image, convert it to RGB channel ordering, and resize
        # it to be a fixed 224x224 pixels, ignoring aspect ratio
        image = cv2.imread(src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(dir)

data = np.array(data)
labels = np.array(labels)
'''

data = np.load('./csv/image_data.npy')
labels = np.load('./csv/image_labels.npy')
print("labels: ", labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print("Prepare Data: ")
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2, stratify=labels, random_state=42)

print("Prepare Gen: ")
# initialize the training data augmentation object
training = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# initialize the validation/testing data augmentation object
validation = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
training.mean = mean
validation.mean = mean

# load the ResNet-50 network
baseModel = ResNet50(weights='imagenet', include_top=False,
                     input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

print("Prepare Model: ")
# actual model (headModel + baseModel)
model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()

# freeze baseModel layers
for layer in baseModel.layers:
    layer.trainable = False

print("Prepare Compile: ")
opt = SGD(learning_rate=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['mae', 'accuracy'])

print("Training: ")
history = model.fit(trainX, trainY, batch_size=25, steps_per_epoch=len(trainX) // 32, validation_data=validation.flow(testX, testY),
                    validation_steps=len(testX) // 32,
                    epochs=epochs)

# save model
model.save_weights('./model/video_model_weights.h5')

'''
# save label binarizer
f = open('./model/video_lb2.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

plt.style.use("ggplot")
plt.figure()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plot_loss)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(plot_acc)

print('predicting')

src = './video/hungry/1.mp4'
size = 128
Q = deque(maxlen=size)

cap = cv2.VideoCapture(src)

while(cap.isOpened()):
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
    print(type(frame))

    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    print('results:', results)
    i = np.argmax(results)
    label = lb.classes_[i]

print(label)
'''
