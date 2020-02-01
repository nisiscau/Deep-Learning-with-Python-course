#Let's see which type we pick for which data type
import tensorflow as tf
import keras

#Simple vector data (2D tensors, [samples,features])is usually
#processed with densely connected features
simplenet = keras.models.Sequential()
simplenet.add(keras.layers.Dense(128,activation='relu',input_shape=(512,512)))

#Sequence data (3D tensors, [samples, timesteps, features]) is
#processed by recurrent layers, like LSTM
sequencenet=keras.models.Sequential()
sequencenet.add(keras.layers.LSTM(64,activation='relu'))

 #Image data (4D tensors, [samples, height, width,color_channel]) is
 #processed by convolutional layers,like Conv2D
imagenet=keras.models.Sequential()
imagenet.add(keras.layers.Conv2D(64,kernel_size=(8,8),activation='relu'))
