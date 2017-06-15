import pandas as pd
import numpy as np
import theano
from keras.models import load_model
theano.config.floatX='float32'

df = pd.read_csv('Data/test.csv')
X = np.array(df)
X = X.astype(theano.config.floatX)
X /= 255
X = X.reshape(28,28,28)
X = X[:,np.newaxis,:,:]
model = load_model('Model-05-0.99.hdf5')
predictions = model.predict_classes(X)
print(predictions)