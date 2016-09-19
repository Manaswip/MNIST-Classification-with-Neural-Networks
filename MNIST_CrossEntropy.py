import random
import numpy as np
import pickle

""" This program is implementation of Neural Networks using cross entroyp
cost function, sigmoid activation function and stochastic gradient descent to
train the network

This program incorparates ideas and code from text book on 'Neural Networks and Deep learning' 
and github https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py
from Michael Nielsen 
"""


class Network(object):

	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
#initializing bias and weights with random numbers
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
# zip returns a tuple of corresponding element x, to correspoding element in y
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]


	def feed_forward(self,input):
		for bias,weight in zip(self.biases,self.weights):
			input = sigmoid(np.dot(weight,input)+bias)
		return input

	def stochasticGradientDescent(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		if test_data: n_test = len(test_data)
		n_train = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n_train, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test)
			else:
				print "Epoch {0} complete". format(j)

	def update_mini_batch(self,mini_batch,eta):
		gradient_bias = [np.zeros(b.shape) for b in self.biases]
		gradient_weights = [np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			delta_bias,delta_weight = self.backprop(x,y)
			gradient_bias = [gb+db for gb,db in zip(gradient_bias,delta_bias)]
			gradient_weights = [gw+dw for gw,dw in zip(gradient_weights,delta_weight)]

		self.biases = [b-(eta/len(mini_batch))*gb for b,gb in zip (self.biases,gradient_bias)]
		self.weights = [w-(eta/len(mini_batch))*gw for w,gw in zip(self.weights,gradient_weights)]

        
	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feed_forward(x)),y) for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)

	def cost_derivative(self,activation,y):
		return (activation-y)

	def backprop(self,x,y):

		derivative_bias = [np.zeros(b.shape) for b in self.biases]
		derivative_weight = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		z_values = []

		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation)+b
			z_values.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = self.cost_derivative(activations[-1],y) * sigmoid_derivate(z_values[-1])
		derivative_bias[-1] = delta
		derivative_weight[-1] = np.dot(delta, activations[-2].transpose())

		for i in xrange(2,self.num_layers):

			z = z_values[-i]
			delta = np.dot(self.weights[-i+1].transpose(),delta) * sigmoid_derivate(z)
			derivative_bias[-i] = delta
			derivative_weight[-i] = np.dot(delta,activations[-i-1].transpose())
		return derivative_bias,derivative_weight


def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivate(z):
	return sigmoid(z) * (1 - sigmoid(z))

			


	

