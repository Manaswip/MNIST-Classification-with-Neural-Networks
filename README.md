<html>
<body>
<h2> MNIST classification with Neural Networks </h2>
MNIST is a large database of handwritten digits. In this project MNIST dataset is trained using deep convolutional nets. Code supports different kind of layer types( Full connected layer, convoluations layer, max pooling layer, softmax layer) and different activation functions (sigmoid, rectified linear units, tanh)

Code is built using Theano library so, this code can be run either on CPU or GPU, set GPU to true to run on GPU and set GPU to false to run on CPU

This program incorparates ideas and code from text book on <a href='http://neuralnetworksanddeeplearning.com/index.html'> Neural Networks and Deep learning from Michael Nielsen </a> and <a href='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src'>Michael Nielsen's github</a> 

<h3> Project Layout </h3>
<p>
<b>MNISTLoadData.py:</b> Unpacks data from the package and returns training,validation, and testing data </br>
<b>MNISTTraining_Theano.py:</b> Implementation of neural network using theano giving an advantage of running the code either on CPU/GPU. In addition to that this code supports different cost functions and activation functions </br>
<b>MNIST_CrossEntropy.py:</b> Implementation of neural network using cross entropy cost function, sigmoid activation function, and stochastic gradient descent </br>
<b>MNIST_QuadraticAndCrossEntropy.py:</b> Implementation of neural network using quadratic cost function, sigmoid activation function, and stochastic gradient descent
<h3> Sample code to run the program </h3>
import cv2 </br> 
import MNISTTraining </br> 
from MNISTTraining import Network </br> 
from MNISTTraining import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer </br> 
training_data, validation_data, test_data = MNISTTraining.load_data_shared() </br> 
mini_batch_size = 10 </br> 
net = Network([ </br>
   &emsp;  &emsp;&emsp;&emsp;  ConvPoolLayer(image_shape=(mini_batch_size, 1, 32, 32), filter_shape=(20, 1, 5, 5),  </br>
    &emsp;  &emsp;&emsp;&emsp;                poolsize=(2, 2)), FullyConnectedLayer(n_in=20*14*14, n_out=100), </br>
     &emsp; &emsp;&emsp;&emsp;                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size) </br> 
net.SGD(training_data, 60, mini_batch_size, 0.1,validation_data, test_data)   </br>

</p>
</body>
</html>
