from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn import cross_validation
from keras.optimizers import SGD, RMSprop, Adam
from keras.datasets import mnist
from quiver_engine import server
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import theano
import os
# define the ConvNet
class LeNet:
	@staticmethod
	def build(input_shape,classes):
		model = Sequential()
		# CONV => RELU => POOL
		model.add(Conv2D(20,kernel_size=5,padding='same',input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		# CONV -> RELU -> POOL
		model.add(Conv2D(50,kernel_size=5,border_mode='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		# Flatten => Relu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation('relu'))
		model.add(Dense(classes))
		model.add(Activation('softmax'))
		return model

# network and training
NB_EPOCH = 15
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28,28 
NB_CLASSES = 10 # NUmber of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
MODEL_DIR = '\home\mayank'
K.set_image_dim_ordering('th')

df = pd.read_csv('Data/main_train.csv')
# Seperate features and labels
X = df.drop('label',1)
y = df['label']
X = np.array(X)
y = np.array(y)
X = X.astype(theano.config.floatX)
y= y.astype(theano.config.floatX)
X = X.reshape(X.shape[0],28,28)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2) 
X_train /= 255
X_test /= 255
X_train = X_train[:,np.newaxis,:,:]
X_test = X_test[:,np.newaxis,:,:]
print(X_train.shape[0],'train_smaples')
print(X_test.shape[0],'test_smaples')

#class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train,NB_CLASSES)
y_test = np_utils.to_categorical(y_test,NB_CLASSES)

# initialize the optimizer and model
model = LeNet.build(input_shape=INPUT_SHAPE,classes=NB_CLASSES)
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
checkpoint = ModelCheckpoint('Model-{epoch:02d}-{val_acc:.2f}.hdf5',verbose=VERBOSE,monitor='val_acc',save_best_only=True)
history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT,callbacks=[checkpoint])
score = model.evaluate(X_test,y_test,verbose=VERBOSE)
print('Test score : ',score[0])
print('Test accuracy : ',score[1])
print(history.history.keys())

#plot the graphs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='upper_left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper_left')
plt.show()

