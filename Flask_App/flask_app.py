from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
from keras.models import load_model
import numpy as np
import os
import re

app = Flask(__name__)
model = load_model('model/model-05-0.99.hdf5')

def convertImage(imgData):
	imgstr = re.search(r'base64,(.*)',imgData).group(1)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))

@app.route('/')
def index():
	return render_template('main.html')


@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)
	x  = imread('output.png',mode='L')
	x = np.invert(x)
	x = x.astype('float32')
	x /= 255
	x = imresize(x,(28,28))
	x = x.reshape(1,1,28,28)
	prediction = model.predict_classes(x)
	response = np.array_str(prediction[0])
	return response


if __name__ == '__main__':
	app.run(debug=True)