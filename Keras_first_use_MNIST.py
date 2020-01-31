import tensorflow as tf
import keras

from keras.datasets import mnist
from keras import layers
from keras.models import Sequential

(train_data,train_labels),(test_data,test_labels)=mnist.load_data()



neuralnet=Sequential()
neuralnet.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
neuralnet.add(layers.Dense(10,activation='softmax'))
