import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
#num_samples, img_features_grid = 100 , 256
#num_user_feature = 10 + 2
#xtrain = np.random.rand(num_samples,
#                        img_features_grid)

x = []
y = []
all_files = os.listdir('../data/samples')
for f in all_files[:250]:
    with open('../data/samples/'+f, 'rb') as handle:
        output_dict = pickle.load(handle)
        x.append(output_dict['x'])
        y.append(output_dict['y'])

print('file loaded')
x = np.array(x)
x = x.reshape(x.shape[0]*x.shape[1],-1)
print(x.shape)
y = np.array(y)
y = y.reshape(y.shape[0]*y.shape[1],-1)
print(y.shape)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

#ytrain = np.random.rand(num_samples,
#                        num_user_feature)

#xtest = np.random.rand(num_samples,
#                        img_features_grid)

#ytest = np.random.rand(num_samples,
#                        num_user_feature)


def fit_model(xtrain, ytrain, xtest, ytest,
              hidden_dim1=128,hidden_dim2=64,drop_out=0.75,optimizer='adam',loss='huber',
              batch_size=32,epochs=3,
              PLOT=True, name = 'model', save_model=False):
    #define model
    num_user_feature = ytrain.shape[1]
    input_user = layers.Input(shape=(xtrain.shape[1]), name='input_layer')

    dense1 = layers.Dense(hidden_dim1, activation='relu', name='dense1')(input_user)  #(dense1)
    dense2 = layers.Dense(hidden_dim2, activation='relu',  name='dense2')(dense1)     #(dense2)
    dpout = layers.Dropout(drop_out, name='dropout')(dense2)
    output = layers.Dense(num_user_feature, activation='relu', name = 'final_layer')(dpout)


    #INITIALIZE THE MODEL AND COMPILE IO
    model = Model(inputs  = input_user,
                  outputs = [output],
                  name = 'USER_PREFERENCE_REGRESSION')


    model.compile(loss={'final_layer': loss,
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

hist, model = fit_model(xtrain, ytrain, xtest, ytest,
              hidden_dim1=32,hidden_dim2=16,drop_out=0.25,optimizer='adam',loss='huber',
              batch_size=32,epochs=300,
              PLOT=True, name = 'model', save_model=False)

