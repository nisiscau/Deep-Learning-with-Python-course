#Importing relevant libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras import layers
from keras.models import Sequential

#Downloading the data
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()

#Creating the shell for our neural net
neuralnet=Sequential()
#Creating the hidden layer:
    #Dense indicates that each neuron of the layer is connected
    #to every example of the previous layer
    #'relu' stands for rectified linear unit (relu(x)=max(0,x))
    #We must also define the expected shape of the inputs
neuralnet.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))

#Creating the output layer: here, we must only define the output shape,
#namely the number of categories into which we want to classify the data

#The activation function, softmax, returns an array of 10 probability
#scores for each category (all probabilities summing to 1)

neuralnet.add(layers.Dense(10,activation='softmax'))

#Let's see the first image in the dataset we're given
plt.imshow(train_data[0], cmap='gray')
plt.show()

#Compilation step: we need 3 equipments for our neural net:
#   a loss function (to compute how far away the result given by
#                   the network is from the actual output)
#   a optimizer (algorithm to improve the network after
#                 it has begun to learn)
#   a metric (the paramater that we want to optimize
#             (in this case, we only care about the accuracy)

neuralnet.compile(optimizer='rmsprop',\
                 loss='categorical_crossentropy',\
                 metrics=['accuracy'])

#Previously, our training data was a 3D tensor of shape (60000,28,28)
#let's reshape it and change the data type so that each pixel is
#a float

train_data=train_data.reshape((60000,28*28))
train_data=train_data.astype('float32')/255

test_data=test_data.reshape((10000,28*28))
test_data=test_data.astype('float32')/255

#let's categorize the labels, i.e. transform them into column vectors
#rather than floats between 0 and 10, with each row having a 1 if the
#number shown is the row number

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#we are all set to train the model! In Keras, the fit method is used
# to fit the data to the neural network

neuralnet.fit(train_data, train_labels, epochs=5, batch_size=128)

#finally, we test the accuracy of the neural net on the test data
test_loss, test_acc = neuralnet.evaluate(test_data, test_labels)
print('The loss and accuracy of the neural net on the test data set are',
      ' respectively {} and {}'.format(test_loss,test_acc))
