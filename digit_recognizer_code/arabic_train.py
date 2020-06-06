#!/usr/bin/env python

import sys

# sys.argv is a list of string where each string is the argument you used after calling python
# the 0th argument will always be your .py file, so you actual input files will be 1:

if len(sys.argv) != 2:
    print ("Usage: base_train.py train.csv")

train_path = sys.argv[1]
print(train_path)

import numpy as np
import pandas as pd
# Models
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.callbacks import ModelCheckpoint

# Data preparation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Plotting and display
from IPython.display import display
from matplotlib import pyplot as plt


np.random.seed(0)

train = pd.read_csv(train_path)

# Scale the image pixel values from 0-255 to 0-1 range so the neural net can converge faster
train.iloc[:, 1:] = train.iloc[:, 1:] / 255

train, test = train_test_split(train, test_size=0.25)

# Separate the label from the data
train_X = train.iloc[:, 1:].values
train_y = train.iloc[:, 0].values.reshape(-1,1)
test_X = test.iloc[:, 1:].values
test_y = test.iloc[:, 0].values.reshape(-1,1)

# function to One-hot encode the categorical values
def one_hot_encode_categories(y):
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    y_one_hot = pd.DataFrame(encoder.fit_transform(y), columns=encoder.get_feature_names())
        
    return y_one_hot, encoder

# One hot encode y
train_y_onehot, encoder = one_hot_encode_categories(train_y)
train_y_onehot.columns = pd.DataFrame(train_y_onehot.columns)[0].apply(lambda x: x[-1])

test_y_onehot, encoder = one_hot_encode_categories(test_y)
test_y_onehot.columns = pd.DataFrame(test_y_onehot.columns)[0].apply(lambda x: x[-1])


# reshape the train digits into 28x28 entries
# so we can plot those pixels
train_X_reshaped = train_X.reshape(-1,28,28)

def plot_sample_images(X, y, images_to_show=10, random=True):

    fig = plt.figure(1)

    images_to_show = min(X.shape[0], images_to_show)

    # Set the canvas based on the numer of images
    fig.set_size_inches(18.5, images_to_show * 0.3)

    # Generate random integers (non repeating)
    if random == True:
        idx = np.random.choice(range(X.shape[0]), images_to_show, replace=False)
    else:
        idx = np.arange(images_to_show)
        
    # Print the images with labels
    for i in range(images_to_show):
        plt.subplot(images_to_show/10 + 1, 10, i+1)
        plt.title(str(y[idx[i]]))
        plt.imshow(X[idx[i], :, :], cmap='Greys')


plot_sample_images(train_X_reshaped, train_y)


def model_definition():
    # Define a simple model in Keras
    model = Sequential()

    # Add layers to the model

    # Add convolutional layer
    model.add(Conv2D(50, kernel_size=(5,5), input_shape=(28,28,1)))

    # Add ReLu activation function
    model.add(Activation('relu'))

    # Add dropout layer for generalization
    model.add(Dropout(0.05))

    # Add maxpool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    # Add batch normalization to help learning and avoid vanishing or exploding gradient
    model.add(BatchNormalization())

    # Add convolutional layer
    model.add(Conv2D(50, kernel_size=(3,3)))

    # Add ReLu activation function
    model.add(Activation('relu'))

    # Add dropout layer for generalization
    model.add(Dropout(0.05))

    # Maxpool layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    # Add batch normalization to help learning and avoid vanishing or exploding gradient
    model.add(BatchNormalization())

    # Add flatten layer to get 1d data for dense layer
    model.add(Flatten())

    # Dense layer
    model.add(Dense(100, input_dim=650))
    
    # Add ReLu activation function
    model.add(Activation('relu'))

    # Dense layer
    model.add(Dense(10))
    
    # Add sigmoid activation function to get values beteween 0-1
    model.add(Activation('softmax'))
    
    return model


model = model_definition()
# Compile the model
model.compile(loss=categorical_crossentropy, 
              optimizer=Adam(lr=0.0005), 
              metrics=["categorical_accuracy"])

# reshape data for model input
train_X_reshaped = train_X.reshape(-1,28,28,1)
test_X_reshaped = test_X.reshape(-1,28,28,1)

# checkpoint
filepath="./arabic.model.best.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_categorical_accuracy', # this must be the same string as a metric from your model training
                             verbose=1, 
                             save_best_only=True, 
                             mode='max',
                             save_weights_only=False)
callbacks_list = [checkpoint]

# Let's train the model and also save the best model weights so we can use them later
history = model.fit(train_X_reshaped, train_y_onehot,
                    batch_size=30,
                    validation_data=(test_X_reshaped, test_y_onehot),
                    epochs=20,
                    callbacks = callbacks_list)

pred_y_onehot = model.predict(test_X_reshaped)
# this will return the one-hot encoded version

# transform back to just output 1 class per instance
pred_y = encoder.inverse_transform(pred_y_onehot)[:,0]

# plot hand-written digits in the test set and our prediction
plot_sample_images(test_X_reshaped.reshape(-1, 28, 28), pred_y)


