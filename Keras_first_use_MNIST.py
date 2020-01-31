#importing relevant libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras import layers
from keras.models import Sequential

#downloading the data
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()

#creating the shell for our neural net
neuralnet=Sequential()
#creating the hidden layer:
    #Dense indicates that each neuron of the layer is connected
    #to every example of the previous layer
    #'relu' stands for rectified linear unit (relu(x)=max(0,x))
    #we must also define the expected shape of the inputs
neuralnet.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))

#creating the output layer: here, we must only define the output shape,
#namely the number of categories into which we want to classify the data
neuralnet.add(layers.Dense(10,activation='softmax'))

#let's see the first image of the dataset we're given
plt.imshow(train_data[0], cmap='gray')
plt.show()
