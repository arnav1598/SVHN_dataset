# PART - 1 : Importing the libraries and packages

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
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

# PART - 2 : Data Preprocessing

# Importing all data
y_train = pd.read_csv('train_data.csv')
y_test = pd.read_csv('test_data.csv')

# Sorting data alphabetically

for i in range(y_train.shape[0]):
    y_train.iloc[i,0]=str(y_train.iloc[i,0])+'.png'
for i in range(y_test.shape[0]):
    y_test.iloc[i,0]=str(y_test.iloc[i,0])+'.png'
y_train = y_train.sort_values('img_name')
y_test = y_test.sort_values('img_name')
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Storing bbox and label data in different variables

bbox_train = y_train.iloc[:, 2:6].values
bbox_test = y_test.iloc[:, 2:6].values
label_train = y_train.iloc[:, 1].values
label_test = y_test.iloc[:, 1].values

# Rescaling bbox data

i=0
for img in os.listdir('train/none/'):
    imgr = mpimg.imread(os.path.join('train/none/', img))
    bbox_train[i][0]=bbox_train[i][0]*(64/imgr.shape[0])
    bbox_train[i][1]=bbox_train[i][1]*(64/imgr.shape[1])
    bbox_train[i][2]=min(64,bbox_train[i][2]*(64/imgr.shape[0]))
    bbox_train[i][3]=min(64,bbox_train[i][3]*(64/imgr.shape[1]))
    i+=1
i=0
for img in os.listdir('test/none/'):
    imgr = mpimg.imread(os.path.join('test/none/', img))
    bbox_test[i][0]=bbox_test[i][0]*(64/imgr.shape[0])
    bbox_test[i][1]=bbox_test[i][1]*(64/imgr.shape[1])
    bbox_test[i][2]=min(64,bbox_test[i][2]*(64/imgr.shape[0]))
    bbox_test[i][3]=min(64,bbox_test[i][3]*(64/imgr.shape[1]))
    i+=1

# Defining functions and generators

def bbox_gen(bbox):
    while True:
        for i in range(0,bbox.shape[0],32):
            n = []
            n = np.resize(n, (0, 4))
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
            n = np.resize(n, (0,1,11))
            for j in range(i, i+32):
                if (j<labels.shape[0]):
                    n=np.append(n, labels[j:j+1, k-1:k, :], axis=0)
            n = np.squeeze(n)
            yield n

# Hot-encoding labels

label_train = process_labels(label_train)
label_test = process_labels(label_test)

# Creating all image generators

datagen = ImageDataGenerator(rescale = 1./255)
datagen2 = ImageDataGenerator(rescale = 1./255,
                              shear_range = 0.1,
                              zoom_range = 0.1,
                              vertical_flip = True)

training_set_d = datagen.flow_from_directory('train/',
                                             target_size = (64, 64),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_d = datagen.flow_from_directory('test/',
                                         target_size = (64, 64),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)

training_set_c_1 = datagen2.flow_from_directory('train_cr/',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_c_1 = datagen2.flow_from_directory('test_cr/',
                                         target_size = (32, 32),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)
training_set_c_2 = datagen2.flow_from_directory('train_cr/',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_c_2 = datagen2.flow_from_directory('test_cr/',
                                         target_size = (32, 32),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)
training_set_c_3 = datagen2.flow_from_directory('train_cr/',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_c_3 = datagen2.flow_from_directory('test_cr/',
                                         target_size = (32, 32),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)
training_set_c_4 = datagen2.flow_from_directory('train_cr/',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_c_4 = datagen2.flow_from_directory('test_cr/',
                                         target_size = (32, 32),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)
training_set_c_5 = datagen2.flow_from_directory('train_cr/',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_c_5 = datagen2.flow_from_directory('test_cr/',
                                         target_size = (32, 32),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)
training_set_c_6 = datagen2.flow_from_directory('train_cr/',
                                             target_size = (32, 32),
                                             batch_size = 32,
                                             shuffle = False,
                                             class_mode = None)
test_set_c_6 = datagen2.flow_from_directory('test_cr/',
                                         target_size = (32, 32),
                                         batch_size = 32,
                                         shuffle = False,
                                         class_mode = None)

# Creating bbox generators

bbox_train_g=bbox_gen(bbox_train)
bbox_test_g=bbox_gen(bbox_test)

# Creating label generators

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

# Zipping

training_set_d = zip(training_set_d, bbox_train_g)
test_set_d = zip(test_set_d, bbox_test_g)

training_set_c_1 = zip(training_set_c_1, label_train_1)
training_set_c_2 = zip(training_set_c_2, label_train_2)
training_set_c_3 = zip(training_set_c_3, label_train_3)
training_set_c_4 = zip(training_set_c_4, label_train_4)
training_set_c_5 = zip(training_set_c_5, label_train_5)
training_set_c_6 = zip(training_set_c_6, label_train_6)

test_set_c_1 = zip(test_set_c_1, label_test_1)
test_set_c_2 = zip(test_set_c_2, label_test_2)
test_set_c_3 = zip(test_set_c_3, label_test_3)
test_set_c_4 = zip(test_set_c_4, label_test_4)
test_set_c_5 = zip(test_set_c_5, label_test_5)
test_set_c_6 = zip(test_set_c_6, label_test_6)

# PART - 3 : Training

# Initialising the CNN for DETECTION

detector = Sequential()
detector.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation='relu'))
detector.add(BatchNormalization())
detector.add(Conv2D(32, (3, 3), activation='relu'))
detector.add(BatchNormalization())
detector.add(Dropout(0.1))
detector.add(MaxPooling2D(pool_size = (2, 2)))
detector.add(Conv2D(64, (3, 3), activation='relu'))
detector.add(BatchNormalization())
detector.add(Conv2D(64, (3, 3), activation='relu'))
detector.add(BatchNormalization())
detector.add(Dropout(0.2))
detector.add(MaxPooling2D(pool_size = (2, 2)))
detector.add(Flatten())
detector.add(Dense(units = 1024, activation='relu'))
detector.add(BatchNormalization())
detector.add(Dropout(0.2))
detector.add(Dense(units = 512, activation='relu'))
detector.add(BatchNormalization())
detector.add(Dropout(0.1))
detector.add(Dense(units = 4))

# Compiling the CNN

detector.compile(optimizer = 'adam', loss = 'mse')

# Fitting the CNN to the images

detector.fit_generator(generator=training_set_d,
                       steps_per_epoch=33402/32,
                       epochs=50,
                       validation_data=test_set_d,
                       validation_steps=13068/32)

# Initialising the CNN for CLASSIFICATION

classifier_1 = Sequential()
classifier_1.add(Conv2D(16, (3, 3), input_shape = (32, 32, 3), activation='relu'))
classifier_1.add(BatchNormalization())
classifier_1.add(Conv2D(32, (3, 3), activation='relu'))
classifier_1.add(BatchNormalization())
classifier_1.add(Dropout(0.1))
classifier_1.add(MaxPooling2D(pool_size = (2, 2)))
classifier_1.add(Conv2D(64, (3, 3), activation='relu'))
classifier_1.add(BatchNormalization())
classifier_1.add(Conv2D(64, (3, 3), activation='relu'))
classifier_1.add(BatchNormalization())
classifier_1.add(Dropout(0.2))
classifier_1.add(MaxPooling2D(pool_size = (2, 2)))
classifier_1.add(Flatten())
classifier_1.add(Dense(units = 512, activation='relu'))
classifier_1.add(BatchNormalization())
classifier_1.add(Dropout(0.1))
classifier_1.add(Dense(units = 512, activation='relu'))
classifier_1.add(BatchNormalization())
classifier_1.add(Dropout(0.1))
classifier_1.add(Dense(units = 11, activation = 'softmax'))

classifier_2 = Sequential()
classifier_2.add(Conv2D(16, (3, 3), input_shape = (32, 32, 3), activation='relu'))
classifier_2.add(BatchNormalization())
classifier_2.add(Conv2D(32, (3, 3), activation='relu'))
classifier_2.add(BatchNormalization())
classifier_2.add(Dropout(0.1))
classifier_2.add(MaxPooling2D(pool_size = (2, 2)))
classifier_2.add(Conv2D(64, (3, 3), activation='relu'))
classifier_2.add(BatchNormalization())
classifier_2.add(Conv2D(64, (3, 3), activation='relu'))
classifier_2.add(BatchNormalization())
classifier_2.add(Dropout(0.2))
classifier_2.add(MaxPooling2D(pool_size = (2, 2)))
classifier_2.add(Flatten())
classifier_2.add(Dense(units = 512, activation='relu'))
classifier_2.add(BatchNormalization())
classifier_2.add(Dropout(0.1))
classifier_2.add(Dense(units = 512, activation='relu'))
classifier_2.add(BatchNormalization())
classifier_2.add(Dropout(0.1))
classifier_2.add(Dense(units = 11, activation = 'softmax'))

classifier_3 = Sequential()
classifier_3.add(Conv2D(16, (3, 3), input_shape = (32, 32, 3), activation='relu'))
classifier_3.add(BatchNormalization())
classifier_3.add(Conv2D(32, (3, 3), activation='relu'))
classifier_3.add(BatchNormalization())
classifier_3.add(Dropout(0.1))
classifier_3.add(MaxPooling2D(pool_size = (2, 2)))
classifier_3.add(Conv2D(64, (3, 3), activation='relu'))
classifier_3.add(BatchNormalization())
classifier_3.add(Conv2D(64, (3, 3), activation='relu'))
classifier_3.add(BatchNormalization())
classifier_3.add(Dropout(0.2))
classifier_3.add(MaxPooling2D(pool_size = (2, 2)))
classifier_3.add(Flatten())
classifier_3.add(Dense(units = 512, activation='relu'))
classifier_3.add(BatchNormalization())
classifier_3.add(Dropout(0.1))
classifier_3.add(Dense(units = 512, activation='relu'))
classifier_3.add(BatchNormalization())
classifier_3.add(Dropout(0.1))
classifier_3.add(Dense(units = 11, activation = 'softmax'))

classifier_4 = Sequential()
classifier_4.add(Conv2D(16, (3, 3), input_shape = (32, 32, 3), activation='relu'))
classifier_4.add(BatchNormalization())
classifier_4.add(Conv2D(32, (3, 3), activation='relu'))
classifier_4.add(BatchNormalization())
classifier_4.add(Dropout(0.1))
classifier_4.add(MaxPooling2D(pool_size = (2, 2)))
classifier_4.add(Conv2D(64, (3, 3), activation='relu'))
classifier_4.add(BatchNormalization())
classifier_4.add(Conv2D(64, (3, 3), activation='relu'))
classifier_4.add(BatchNormalization())
classifier_4.add(Dropout(0.2))
classifier_4.add(MaxPooling2D(pool_size = (2, 2)))
classifier_4.add(Flatten())
classifier_4.add(Dense(units = 512, activation='relu'))
classifier_4.add(BatchNormalization())
classifier_4.add(Dropout(0.1))
classifier_4.add(Dense(units = 512, activation='relu'))
classifier_4.add(BatchNormalization())
classifier_4.add(Dropout(0.1))
classifier_4.add(Dense(units = 11, activation = 'softmax'))

# Compiling the CNN

classifier_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
classifier_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
classifier_3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
classifier_4.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

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

# PART - 4 : Testing on a single image

# Loading and preprocessing test image

test_image = Image.open('roomnum.jpg')
test_image = test_image.resize((64,64))
test_image = np.array(test_image).astype(float)
test_image = np.expand_dims(test_image, axis = 0)
test_image /= 255.0

# Predicting the bounding box

bbox = detector.predict(test_image)
bbox[0,2]=min(bbox[0,2],64)
bbox[0,3]=min(bbox[0,3],64)
bbox = bbox.astype(int)

# Crop and Resize using bbox

test_image = np.squeeze(test_image)
test_image = Image.fromarray(np.uint8(test_image*255))
test_image = test_image.crop((bbox[0,1],bbox[0,0],bbox[0,3],bbox[0,2]))
test_image = test_image.resize((32,32))
test_image = np.array(test_image).astype(float)
test_image = np.expand_dims(test_image, axis = 0)
test_image /= 255.0

# Predicting the digits

num=0
d1 = np.argmax(classifier_1.predict(test_image))+1
d2 = np.argmax(classifier_2.predict(test_image))+1
d3 = np.argmax(classifier_3.predict(test_image))+1
d4 = np.argmax(classifier_4.predict(test_image))+1

if d1<11:
    num = num*10 + d1%10
if d2<11:
    num = num*10 + d2%10
if d3<11:
    num = num*10 + d3%10
if d4<11:
    num = num*10 + d4%10

print(num)
