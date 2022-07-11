import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os


num_samples, img_features_grid = 100 , 256
num_user_feature = 10 + 2

xtrain = np.random.rand(num_samples,
                        img_features_grid)

ytrain = np.random.rand(num_samples,
                        num_user_feature)

xtest = np.random.rand(num_samples,
                        img_features_grid)

ytest = np.random.rand(num_samples,
                        num_user_feature)


def fit_model(xtrain, ytrain, xtest, ytest,
              hidden_dim1=128,hidden_dim2=64,drop_out=0.75,optimizer='adam',loss='huber',
              batch_size=32,epochs=3,
              PLOT=True, name = 'model'):
    #define model
    input_user = layers.Input(shape=(xtrain.shape[1]), name='input_layer')

    dense1 = layers.Dense(hidden_dim1, activation='relu', name='dense1')(input_user) #(dense1)
    dense2 = layers.Dense(hidden_dim2, activation='relu',  name='dense2')(dense1)     #(dense2)
    dpout = layers.Dropout(drop_out, name='dropout')(dense2)
    output = layers.Dense(num_user_feature, activation='relu', name = 'final_layer')(dpout)



    #INITIALIZE THE MODEL AND COMPILE IO
    model = Model(inputs  = input_user,
                  outputs = [output],
                  name = 'USER_PREFERENCE_REGRESSION')


    model.compile(loss={'final_layer': 'huber',
                       },
                  #loss_weights={},
                  optimizer=optimizer
                  )

    #MODEL SUMMARY PRINT
    model.summary()


    #Fit Data
    history = model.fit(xtrain, {'final_layer': ytrain},

                        validation_data= (xtest,
                                          {'final_layer': ytest}),


                        batch_size=batch_size,
                        epochs=epochs,

                        verbose=2,
                        shuffle=True
                        )
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



def make_predictions(model_path, samples_to_be_predicted, batch_size=32):
    # Load the model
    model = keras.models.load_model(filepath, compile = True)
    #make predictions
    predictions = model.predict(samples_to_be_predicted)

    return predictions

