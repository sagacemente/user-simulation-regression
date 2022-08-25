import time
import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers

from keras.backend import sigmoid
# Getting the Custom object and updating them
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Conv1D, LSTM, Dense, Flatten, Dropout, MaxPooling2D,MaxPooling1D, Activation, TimeDistributed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from data_generator.grid_constructor import *


start = time.time()
TEST_SIZE = 0.1
#num_samples, img_features_grid = 100 , 256
#num_user_feature = 10 + 2
#xtrain = np.random.rand(num_samples,
#                        img_features_grid)

x = []
y = []
#all_files = os.listdir('../data/samples')
all_files = os.listdir('../data/samples_clear')
for f in all_files:
    with open('../data/samples_clear/'+f, 'rb') as handle:
        output_dict = pickle.load(handle)
        x.append(output_dict['x'])
        y.append(output_dict['y'])

print('file loaded')
x = np.array(x)
print('x pre shape', x.shape)
#x = average_input(x)
#x = x.reshape(x.shape[0], x.shape[1],-1)
#print('x post shape', x.shape)

y = np.array(y)
#y = y.reshape(y.shape[0]*y.shape[1],-1)
y = y[:, 0, :-2]
print('y shape', y.shape)

#split train test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
for i in [xtrain, xtest, ytrain, ytest]:
    print('set shape\t',i.shape)


model = Sequential()
model.add(Conv1D(16, kernel_size=8, padding = 'same', input_shape=(30,64)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.2))

model.add(Conv1D(32, kernel_size=4, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Flatten()))  #https://keras.io/api/layers/recurrent_layers/time_distributed/
model.add(LSTM(8))
model.add(Dense(7, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
history = model.fit(X, y)
