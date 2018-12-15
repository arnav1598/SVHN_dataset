# Importing the Keras libraries and packages

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

# Initialising the CNN

detector = Sequential()
detector.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
detector.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
detector.add(Dropout(0.2))
detector.add(MaxPooling2D(pool_size = (2, 2)))
detector.add(Conv2D(64, (3, 3), activation = 'relu'))
detector.add(Conv2D(64, (3, 3), activation = 'relu'))
detector.add(Dropout(0.2))
detector.add(MaxPooling2D(pool_size = (2, 2)))
detector.add(Conv2D(128, (3, 3), activation = 'relu'))
detector.add(Conv2D(128, (3, 3), activation = 'relu'))
detector.add(Dropout(0.2))
detector.add(Flatten())
detector.add(Dense(units = 1024, activation = 'relu'))
detector.add(Dropout(0.2))
detector.add(Dense(units = 1024, activation = 'relu'))
detector.add(Dense(units = 4))

# Compiling the CNN

detector.compile(optimizer = 'adadelta', loss = 'mse')

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

y_train = pd.read_csv('train_data.csv')
y_test = pd.read_csv('test_data.csv')
bbox_train = y_train.iloc[:, 2:6].values
bbox_test = y_test.iloc[:, 2:6].values

i=0
for img in os.listdir('train/none/'):
    img = mpimg.imread(os.path.join('train/none/', img))
    bbox_train[i][0]=bbox_train[i][0]*(64/img.shape[0])
    bbox_train[i][1]=bbox_train[i][1]*(64/img.shape[1])
    bbox_train[i][2]=bbox_train[i][2]*(64/img.shape[0])
    bbox_train[i][3]=bbox_train[i][3]*(64/img.shape[1])
    i+=1
i=0
for img in os.listdir('test/none/'):
    img = mpimg.imread(os.path.join('test/none/', img))
    bbox_test[i][0]=bbox_test[i][0]*(64/img.shape[0])
    bbox_test[i][1]=bbox_test[i][1]*(64/img.shape[1])
    bbox_test[i][2]=bbox_test[i][2]*(64/img.shape[0])
    bbox_test[i][3]=bbox_test[i][3]*(64/img.shape[1])
    i+=1

training_set = train_datagen.flow_from_directory('train/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = None)

test_set = test_datagen.flow_from_directory('test/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = None)

def bbox_train_gen():
    while True:
        for i in range(0,33402,32):
            n = []
            n=np.resize(n, (0, 4))
            for j in range(i, i+32):
                if (j<33402):
                    n=np.append(n, bbox_train[j:j+1, :], axis=0)
            yield n

def bbox_test_gen():
    while True:
        for i in range(0,13068,32):
            n = []
            n=np.resize(n, (0, 4))
            for j in range(i, i+32):
                if (j<13068):
                    n=np.append(n, bbox_test[j:j+1, :], axis=0)
            yield n

ax=bbox_train_gen()
ay=bbox_test_gen()

training_set = zip(training_set, ax)
test_set = zip(test_set, ay)

detector.fit_generator(generator=training_set,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set,
                       validation_steps=13068/32)