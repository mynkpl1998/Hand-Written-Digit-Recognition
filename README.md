# Hand Written Digit Recognition
Detecting digit on the images using Convolution Neural Networks  

<img src="https://raw.githubusercontent.com/mynkpl1998/Hand-Written-Digit-Recognition/master/demo.png" width=800 height=400>


* The neural network network is trained on the grayscale images provided by kaggle form MNIST competition. "Data" folder    contains the data to train the images. 

* "conv_net.py" script builds the convolutional network and dumps it in .hdf5 file, which reduces the overhead of training the model again.

* **Flask App folder contains the code for the app implmented [here](http://mynkpl1998.pythonanywhere.com)**

# Artitecture of artificial neural network
  
  1. 2D Convoultion
  2. Relu Activation
  3. 2D MAX Pooling
  4. 2D Convoultion
  5. Relu Activation
  6. 2D MAX Pooling
  7. Dense layer with ouput dim of 500
  8. Relu Activation
  9. Ouput Layer with dim of 10
  10. Softmax Activation
  
# Dependencies
  * Tensorflow
  * Keras
  * Theano
  * h5py
  * Flask


# Model Accuracy on Validation Data : 99%
# Modely Accuracy on Training Data : 99.164%
