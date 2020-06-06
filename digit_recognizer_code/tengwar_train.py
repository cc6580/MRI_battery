#!/usr/bin/env python

import sys

# sys.argv is a list of string where each string is the argument you used after calling python
# the 0th argument will always be your .py file, so you actual input files will be 1:

if len(sys.argv) != 2:
    print ("Usage: base_train.py train.csv modle.h5")

train_path = sys.argv[1]
model_path = sys.argv[2]

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
from scipy.interpolate import interp2d

# Plotting and display
from IPython.display import display
from matplotlib import pyplot as plt


np.random.seed(0)

train = pd.read_csv(train_path)

train.iloc[:,0] = train.iloc[:,0].apply(lambda x: int(x))

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
train_y_onehot.columns = pd.DataFrame(train_y_onehot.columns)[0].apply(lambda x: x[3:])

test_y_onehot, encoder = one_hot_encode_categories(test_y)
test_y_onehot.columns = pd.DataFrame(test_y_onehot.columns)[0].apply(lambda x: x[3:])

# reshape the train digits into 64x64 entries
# so we can plot those pixels
train_X_reshaped = train_X.reshape(-1,64,64)
test_X_reshaped = test_X.reshape(-1,64,64)

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


plot_sample_images(train_X_reshaped, train_y, images_to_show=30)

# we have to iterpolate and resclae for EACH image
train_X_rescaled = np.empty((train_X.shape[0], 28,28))

for image in range(train_X.shape[0]):
  f = interp2d(x = np.arange(0,64),
               y = np.arange(0,64),
               z = train_X_reshaped[image])
  rescaled_values = f(x = np.arange(0,64,64/28),
                      y = np.arange(0,64,64/28))
  train_X_rescaled[image] = rescaled_values

  plot_sample_images(train_X_rescaled, train_y, images_to_show=30)

  # again, rescale the test set also
# we have to iterpolate and resclae for EACH image
test_X_rescaled = np.empty((test_X.shape[0], 28,28))

for image in range(test_X.shape[0]):
  f = interp2d(x = np.arange(0,64),
               y = np.arange(0,64),
               z = test_X_reshaped[image])
  rescaled_values = f(x = np.arange(0,64,64/28),
                      y = np.arange(0,64,64/28))
  test_X_rescaled[image] = rescaled_values

base_model = load_model(model_path)

# reshape data for model input
train_X_reshaped = train_X_rescaled.reshape(-1,28,28,1)
test_X_reshaped = test_X_rescaled.reshape(-1,28,28,1)

# first, let's pop out the last dense layers 
for i in range(4):
  base_model.pop()

x = base_model.output
x=Dense(200,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(100,activation='relu')(x) #dense layer 2
preds=Dense(12,activation='softmax')(x) #final layer with softmax activation for our 12 output classes

tengwar_model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

for layer in tengwar_model.layers[:-3]:
    layer.trainable=False
for layer in tengwar_model.layers[-3:]:
    layer.trainable=True

# Compile the model
tengwar_model.compile(loss=categorical_crossentropy, 
              optimizer=Adam(lr=0.0005), 
              metrics=["categorical_accuracy"])

# checkpoint
filepath="./tengwar.model.best.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_categorical_accuracy', # this must be the same string as a metric from your model training
                             verbose=1, 
                             save_best_only=True, 
                             mode='max',
                             save_weights_only=False)

# # Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

callbacks_list = [checkpoint]

# Let's train the model and also save the best model weights so we can use them later
history = tengwar_model.fit(train_X_reshaped, train_y_onehot,
                    batch_size=30,
                    validation_data=(test_X_reshaped, test_y_onehot),
                    epochs=50,
                    callbacks = callbacks_list)

# load and evaluate the best model
tengwar_model = load_model('./tengwar.model.best.h5')
scores = tengwar_model.evaluate(test_X_reshaped, test_y_onehot, verbose=0)
print("validation %s: %.2f%%" % (tengwar_model.metrics_names[1], scores[1]*100))

pred_y_onehot = tengwar_model.predict(test_X_reshaped)
# this will return the one-hot encoded version

# transform back to just output 1 class
pred_y = encoder.inverse_transform(pred_y_onehot)[:,0]

# plot hand-written digits in the test set and our prediction
plot_sample_images(test_X_reshaped.reshape(-1, 28, 28), pred_y, images_to_show=20)


