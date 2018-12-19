# PART - 1 : Importing the Keras libraries and packages

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
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import crop_to_bounding_box
from tensorflow.image import resize_area
import tensorflow as tf

# PART - 2 : Data Pre-processing

y_train = pd.read_csv('train_data.csv')
y_test = pd.read_csv('test_data.csv')
bbox_train = y_train.iloc[:, 2:6].values
bbox_test = y_test.iloc[:, 2:6].values
label_train = y_train.iloc[:, 1].values
label_test = y_test.iloc[:, 1].values

for img in os.listdir('train/none/'):
    imgr = mpimg.imread(os.path.join('train/none/', img))
    bbox_train[int(img.replace('.png', ''))-1][0]=bbox_train[int(img.replace('.png', ''))-1][0]*(64/imgr.shape[0])
    bbox_train[int(img.replace('.png', ''))-1][1]=bbox_train[int(img.replace('.png', ''))-1][1]*(64/imgr.shape[1])
    bbox_train[int(img.replace('.png', ''))-1][2]=min(64,bbox_train[int(img.replace('.png', ''))-1][2]*(64/imgr.shape[0]))
    bbox_train[int(img.replace('.png', ''))-1][3]=min(64,bbox_train[int(img.replace('.png', ''))-1][3]*(64/imgr.shape[1]))

for img in os.listdir('test/none/'):
    imgr = mpimg.imread(os.path.join('test/none/', img))
    bbox_test[int(img.replace('.png', ''))-1][0]=bbox_test[int(img.replace('.png', ''))-1][0]*(64/imgr.shape[0])
    bbox_test[int(img.replace('.png', ''))-1][1]=bbox_test[int(img.replace('.png', ''))-1][1]*(64/imgr.shape[1])
    bbox_test[int(img.replace('.png', ''))-1][2]=min(64,bbox_test[int(img.replace('.png', ''))-1][2]*(64/imgr.shape[0]))
    bbox_test[int(img.replace('.png', ''))-1][3]=min(64,bbox_test[int(img.replace('.png', ''))-1][3]*(64/imgr.shape[1]))

def bbox_gen(bbox):
    while True:
        for i in range(0,bbox.shape[0],32):
            n = []
            n=np.resize(n, (0, 4))
            for j in range(i, i+32):
                if (j<bbox.shape[0]):
                    n=np.append(n, bbox[j:j+1, :], axis=0)
            yield n

def process_labels(labels):
    temp = []
    temp = np.reshape(temp, (0, 6))
    for label in labels:
        vec = [int(float(x)) for x in label.split('_')]
        for i in range(len(vec), 6):
            vec.append(11)
        vec = np.reshape(vec, (1,6))
        temp = np.append(temp, vec, axis=0)
    labels = np.array(temp)
    temp = []
    for row in labels[:,...]:
        y=np.zeros((6, 11))
        for i in range(6):
            y[i][int(row[i])-1]=1
        temp.append(y)
    labels = np.array(temp)
    return labels

def labels_gen(labels, k):
    while True:
        for i in range(0, labels.shape[0], 32):
            n = []
            for j in range(i, i+32):
                if (j<labels.shape[0]):
                    n.append(labels[j, k-1, :])
            n=np.array(n)
            yield n

def crop_tr(batch):
    i=-1
    sess=tf.Session()
    while (1):
        X,y=next(batch)
        for t in range(X.shape[0]):
            i+=1
            i%=33402
            img = X[t:t+1,:,:,:]
            img = crop_to_bounding_box(img, bbox_train[i,0], bbox_train[i,1], bbox_train[i,2]-bbox_train[i,0], bbox_train[i,3]-bbox_train[i,1])
            img = resize_area(img, (64,64))
            img = sess.run(img)
            X[t,:,:,:]=img
        yield X,y
    sess.close()

def crop_te(batch):
    i=-1
    sess=tf.Session()
    while (1):
        X,y=next(batch)
        for t in range(X.shape[0]):
            i+=1
            i%=13068
            img = X[t:t+1,:,:,:]
            img = crop_to_bounding_box(img, bbox_test[i,0], bbox_test[i,1], bbox_test[i,2]-bbox_test[i,0], bbox_test[i,3]-bbox_test[i,1])
            img = resize_area(img, (64,64))
            img = sess.run(img)
            X[t,:,:,:]=img
        yield X,y
    sess.close()

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = None)

test_set = test_datagen.flow_from_directory('test/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = None)

bbox_train_g=bbox_gen(bbox_train)
bbox_test_g=bbox_gen(bbox_test)

label_train = process_labels(label_train)
label_test = process_labels(label_test)

label_train_1 = labels_gen(label_train, 1)
label_train_2 = labels_gen(label_train, 2)
label_train_3 = labels_gen(label_train, 3)
label_train_4 = labels_gen(label_train, 4)
label_train_5 = labels_gen(label_train, 5)
label_train_6 = labels_gen(label_train, 6)

label_test_1 = labels_gen(label_test, 1)
label_test_2 = labels_gen(label_test, 2)
label_test_3 = labels_gen(label_test, 3)
label_test_4 = labels_gen(label_test, 4)
label_test_5 = labels_gen(label_test, 5)
label_test_6 = labels_gen(label_test, 6)

training_set_d = zip(training_set, bbox_train_g)
test_set_d = zip(test_set, bbox_test_g)

training_set_c_1 = crop_tr(zip(training_set, label_train_1))
training_set_c_2 = crop_tr(zip(training_set, label_train_2))
training_set_c_3 = crop_tr(zip(training_set, label_train_3))
training_set_c_4 = crop_tr(zip(training_set, label_train_4))
training_set_c_5 = crop_tr(zip(training_set, label_train_5))
training_set_c_6 = crop_tr(zip(training_set, label_train_6))

test_set_c_1 = crop_te(zip(test_set, label_test_1))
test_set_c_2 = crop_te(zip(test_set, label_test_2))
test_set_c_3 = crop_te(zip(test_set, label_test_3))
test_set_c_4 = crop_te(zip(test_set, label_test_4))
test_set_c_5 = crop_te(zip(test_set, label_test_5))
test_set_c_6 = crop_te(zip(test_set, label_test_6))

# PART - 3 : Training

# Initialising the CNN for DETECTION

detector = Sequential()
detector.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation='relu'))
detector.add(Dropout(0.2))
detector.add(MaxPooling2D(pool_size = (2, 2)))
detector.add(Conv2D(32, (3, 3), activation='relu'))
detector.add(Dropout(0.2))
detector.add(MaxPooling2D(pool_size = (2, 2)))
detector.add(Flatten())
detector.add(Dense(units = 1024, activation='relu'))
detector.add(BatchNormalization())
detector.add(Dropout(0.2))
detector.add(Dense(units = 512, activation='relu'))
detector.add(Dropout(0.2))
detector.add(Dense(units = 4))

# Compiling the CNN

detector.compile(optimizer = 'adam', loss = 'mse')

# Fitting the CNN to the images

detector.fit_generator(generator=training_set_d,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_d,
                       validation_steps=13068/32)

# Initialising the CNN for CLASSIFICATION

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation='tanh'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='tanh'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='tanh'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 512, activation='tanh'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 512, activation='tanh'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 11, activation = 'softmax'))

classifier_1=classifier
classifier_2=classifier
classifier_3=classifier
classifier_4=classifier
classifier_5=classifier
classifier_6=classifier

# Compiling the CNN

classifier_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
classifier_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
classifier_3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
classifier_4.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
classifier_5.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
classifier_6.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images

classifier_1.fit_generator(generator=training_set_c_1,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_c_1,
                       validation_steps=13068/32)
classifier_2.fit_generator(generator=training_set_c_2,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_c_2,
                       validation_steps=13068/32)
classifier_3.fit_generator(generator=training_set_c_3,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_c_3,
                       validation_steps=13068/32)
classifier_4.fit_generator(generator=training_set_c_4,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_c_4,
                       validation_steps=13068/32)
classifier_5.fit_generator(generator=training_set_c_5,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_c_5,
                       validation_steps=13068/32)
classifier_6.fit_generator(generator=training_set_c_6,
                       steps_per_epoch=33402/32,
                       epochs=25,
                       validation_data=test_set_c_6,
                       validation_steps=13068/32)

# PART - 4 : Testing
