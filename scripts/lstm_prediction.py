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
x = x.reshape(x.shape[0], x.shape[1],-1)
print('x post shape', x.shape)

y = np.array(y)
#y = y.reshape(y.shape[0]*y.shape[1],-1)
y = y[:, 0, :-2]
print('y shape', y.shape)

#split train test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
for i in [xtrain, xtest, ytrain, ytest]:
    print('set shape\t',i.shape)

#ytrain = np.random.rand(num_samples,
#                        num_user_feature)

#xtest = np.random.rand(num_samples,
#                        img_features_grid)

#ytest = np.random.rand(num_samples,
#                        num_user_feature)

# Importing the sigmoid function from
# Keras backend and using it
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
# Below in place of swish you can take any custom key for the name
get_custom_objects().update({'swish': Activation(swish)})

def fit_model(xtrain, ytrain, xtest, ytest,
              hidden_dim1=128,hidden_dim2=64,drop_out=0.75,optimizer='adam',loss='mae',activation ='huber',
              batch_size=10,epochs=3, early_stop_patience = 20,
              PLOT=True, name = 'model', save_model=False):
    #define model

    num_user_feature = ytrain.shape[1]
    input_user = layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]), name='input_layer')

    whole_seq_output, final_memory_state, final_carry_state = layers.LSTM(30, return_sequences=True, return_state=True)(input_user)
    dense1 = layers.Dense(hidden_dim1, activation=activation, name='dense1')(final_memory_state)  #(dense1)
    dpout = layers.Dropout(drop_out, name='dropout')(dense1)
    dense2 = layers.Dense(hidden_dim2, activation=activation,  name='dense2')(dpout)     #(dense2)
    output = layers.Dense(num_user_feature, activation=activation, name = 'final_layer')(dense2)


    #INITIALIZE THE MODEL AND COMPILE IO
    model = Model(inputs  = input_user,
                  outputs = [output],
                  name = 'USER_PREFERENCE_REGRESSION')

    if loss == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError(reduction="auto")
    elif loss == 'mape':
        loss = tf.keras.losses.MeanAbsolutePercentageError(reduction="auto")
    elif loss == 'huber':
        loss = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
    elif loss == 'cosine':
        #Note that it is a number between -1 and 1. When it is a negative number between -1 and 0, 0 indicates orthogonality and
        # values closer to -1 indicate greater similarity.
        # The values closer to 1 indicate greater dissimilarity.
        # This makes it usable as a loss function in a setting where you try to maximize the proximity between predictions and targets.
        loss = tf.keras.losses.CosineSimilarity(axis=-1, reduction="auto")
    elif loss =='mse':
        loss = tf.keras.losses.MeanSquaredError(reduction='auto',name='mean_squared_error')

    model.compile(loss={'final_layer': loss,
                       },
                  #loss_weights={},
                  optimizer=optimizer
                  )

    #MODEL SUMMARY PRINT
    model.summary()

    #CallBacks
    escb = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0.0001, # Minimum change in the monitored quantity to qualify as an improvement
                                            patience=early_stop_patience, #Number of epochs with no improvement after which training will be stopped.
                                            verbose=0,
                                            mode="auto",
                                            baseline=None,
                                            restore_best_weights=True,
                                            )
    #Fit Data
    history = model.fit(xtrain, {'final_layer': ytrain},

                        validation_data= (xtest,
                                          {'final_layer': ytest}),


                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[escb],
                        verbose=2,
                        shuffle=True
                        )
    print('min loss\t', min(history.history['loss']))
    print('min val_loss  ', min(history.history['val_loss']))
    if PLOT == True:
        plt.plot(history.history['loss'],     label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.title('total loss')
        plt.show()
    if save_model == True:
        name_m = os.getcwd() + '\\'+ name
        model.save(name_m)
    return model, history


def compute_loss(loss='mae'):
    if loss == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError(reduction="auto")
    elif loss == 'mape':
        loss = tf.keras.losses.MeanAbsolutePercentageError(reduction="auto")
    elif loss == 'huber':
        loss = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
    elif loss == 'cosine':
        loss = tf.keras.losses.CosineSimilarity(axis=-1, reduction="auto")
    elif loss =='mse':
        loss = tf.keras.losses.MeanSquaredError(reduction='auto',name='mean_squared_error')
    return loss

def make_predictions(model_path, samples_to_be_predicted, batch_size=32):
    # Load the model
    #model = keras.models.load_model(filepath, compile = True)
    #make predictions
    predictions = model.predict(samples_to_be_predicted)

    return predictions

LOSS = 'mse'
model, hist = fit_model(xtrain, ytrain, xtest, ytest,
              hidden_dim1=32,hidden_dim2=16,drop_out=0.5,optimizer='adam',loss=LOSS,activation ='swish',
              batch_size=160,epochs=1000,early_stop_patience = 250,
              PLOT=True, name = 'model', save_model=False)

#2 samples predicitons
predictions = model.predict(xtest[3:6])
all_preds =  model.predict(xtest)
dfpreds = pd.DataFrame(all_preds)
#dfpreds.boxplot()
#plt.show()
dfreal = pd.DataFrame(ytest)
#dfreal.boxplot()
#plt.show()



# make a list of all dataframes
df_list = [dfpreds ,dfreal]

print('prediction\nshape', predictions.shape,'\n',np.around(predictions,2) )
print('actual\nshape', ytest[3:6].shape, '\n',np.around(ytest[3:6], 2))
results = model.evaluate(xtest, ytest, batch_size=15)

avg_baseline_preds = np.mean(ytest, axis=0)

baseline_preds = np.array([0.5]*ytest.size).reshape((ytest.shape[0], -1))
myloss = compute_loss(loss=LOSS)
loss_baseline =  myloss(ytest, avg_baseline_preds).numpy()
loss_avg = myloss(ytest, baseline_preds).numpy()
print('model loss    for test set\t', round(results,4))
print('loss_baseline for test set\t', round(loss_baseline,4))
print('loss AVG pred for test set\t', round(loss_avg,4))
end = time.time()

print('tot time', round((end- start)/60, 2))
