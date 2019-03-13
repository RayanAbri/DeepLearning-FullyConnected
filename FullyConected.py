#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:33:30 2019

@author: rahem
"""

#Similar to the MNIST digit dataset, the Fashion dataset includes:
#
#60,000 training examples
#10,000 testing examples
#10 classes
#28×28 grayscale/single channel images
#The ten fashion class labels include:
#
#1     T-shirt/top  
#2     Trouser/pants
#3     Pullover shirt
#4     Dress
#5     Coat
#6     Sandal
#7     Shirt
#8     Sneaker
#9     Bag
#10    Ankle boot

from keras.datasets import fashion_mnist
from keras.utils import np_utils


def Ciz_history(net):
    history = net.history
    import matplotlib.pyplot as plt
    losses = history['loss']
    accuracies = history['acc']
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.plot(losses)
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.plot(accuracies)


#data 

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

print("Train set Images-> Main Shape :", trainX.shape)
print("Train set Images-> dimension :", trainX.ndim)
print("Train set Images-> data type :", trainX.dtype)

print("Train set labels-> Main Shape :", trainY.shape)
print("Train set labels-> dimension :", trainY.ndim)
print("Train set labels-> data type :", trainY.dtype)


#resim = trainX[1124]
#plt.imshow(resim, cmap='gray') #'binary' barakse color mape gray hastesh

X_train = trainX.reshape(60000,784)
X_test = testX.reshape(10000,784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(trainY)
Y_test = np_utils.to_categorical(testY)

#=======================================================================
#model
from keras.models import Sequential

myModel = Sequential(name='rayan')

#fully connected layers

from keras.layers import Dense


myModel.add(Dense(500, activation='relu', input_shape=(784,),name='Layer0'))
myModel.add(Dense(100, activation='relu',name='Layer1'))
myModel.add(Dense(10, activation='softmax',name='Layer2'))

myModel.summary()


#=========================================================================
#callbacks and plot model

from keras.utils import plot_model
from keras import callbacks 
import keras



plot_model(myModel, to_file='rayan_fullyConnected.png', show_shapes= True )

logger = callbacks.CSVLogger('training.log')


xx = keras.callbacks.TensorBoard(log_dir="/home/rahem/SpyderProjects/CNN")


#=======================================================================
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

myModel.compile(optimizer=SGD(lr=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

networkHistory=myModel.fit(X_train, Y_train, batch_size=128, epochs=10,callbacks=[logger,xx])
#========================================================================


Ciz_history(networkHistory)

# Evaluation

T_LOSS, T_ACC = myModel.evaluate(X_test, Y_test)

test_labels_p = myModel.predict(X_test)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)

#=============================================================================


























































