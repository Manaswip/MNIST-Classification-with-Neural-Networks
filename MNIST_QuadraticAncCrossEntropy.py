import random
import json
import sys
import numpy as np 

""" This program is implementation of Neural Networks with 2 options for cost function
cost entropy and quadatric cost function, sigmoid activation function and stochastic gradient descent to
train

This program incorparates ideas and code from text book on 'Neural Networks and Deep learning' 
and github https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py
from Michael Nielsen 
"""
class CostEntropy(object):
	@staticmethod
	def costFunction(a,y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
	@staticmethod
	def error_delta(z,a,y):
		return a-y

class quadraticCost(object):
	@staticmethod
	def costFunction(a,y):
		return 0.5*np.linalg.norm(a-y)**2
	@staticmethod
	def error_delta(z,a,y):
		return (a-y) * sigmoid_derivate(z)


class Network(object):
	 def __init__(self,sizes,cost=CostEntropy):
	 	self.num_of_layers = len(sizes)
	 	self.sizes = sizes
	 	self.biases = [np.random.randn(b,1) for b in sizes[1:]]
	 	self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])]
	 	self.cost = cost

	 def feedforward(self,x):
	 	for b,w in zip(self.biases,self.weights):
	 		x = sigmoid(np.dot(w,x)+b)
	 	return x

	 def update_mini_batch(self,mini_batch,eta,lmbda,n):
	 	gradient_bias = [np.zeros(b.shape) for b in self.biases]
	 	gradient_weight = [np.zeros(w.shape) for w in self.weights]
	 	for x,y in mini_batch:
	 		delta_gradient_bias,delta_gradient_weight = self.backprop(x,y)
	 		gradient_bias = [db+gb for db,gb in zip(delta_gradient_bias,gradient_bias)]
	 		gradient_weight = [dw+gw for dw,gw in zip(delta_gradient_weight,gradient_weight)]

	 	self.biases = [b - (eta/len(mini_batch))*gb for b,gb in zip(self.biases,gradient_bias)]
	 	self.weights = [(1-(eta*lmbda)/n)*w - (eta/len(mini_batch))*gw for w,gw in zip(self.weights,gradient_weight)]


	 def stochasticGradientDescent(self,training_data,epochs,mini_batch_size,eta,
	 	                           lmbda=0.0,
	 	                           evaluation_data=None,
	 	                           monitor_evaluation_cost=False,
	 	                           monitor_evaluation_accuracy=False,
	 	                           monitor_training_cost=False,
	 	                           monitor_training_accuracy=False):

	 	if evaluation_data: n_eval = len(evaluation_data)
	 	n_train = len(training_data)
	 	training_cost,evaluation_cost = [],[]
	 	training_accuracy,evaluation_accuracy=[],[]
	 	for j in xrange(epochs):
	 		random.shuffle(training_data)
	 		mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n_train,mini_batch_size)]

	 		for mini_batch in mini_batches:
	 			self.update_mini_batch(mini_batch,eta,lmbda,n_train)

	 		print "Epoch %s is complete" %j

	 		if monitor_evaluation_cost:
	 			cost = self.total_cost(evaluation_data,lmbda,convert=True)
	 			evaluation_cost.append(cost)
	 			print "Evaluation cost: {} ".format(cost)

	 		if monitor_evaluation_accuracy:
	 			accuracy = self.accuracy(evaluation_data)
	 			evaluation_accuracy.append(accuracy)
	 			print "Accuracy on evaluation data: {}/{}". format(accuracy,n_eval)

	 		if monitor_training_cost:
	 			cost = self.total_cost(training_data,lmbda)
	 			training_cost.append(cost)
	 			print " Training cost: {} ".format(cost)

	 		if monitor_training_accuracy:
	 			accuracy = self.accuracy(training_data,convert=True)
	 			training_accuracy.append(accuracy)
	 			print "Training accuract: {}/{} ".format(accuracy,n_train)

	 		print

	 	return evaluation_cost,evaluation_accuracy,training_cost,training_accuracy

	 def accuracy(self,data,convert=False):
	 	if convert:
	 		results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in data]
	 	else:
	 		results = [(np.argmax(self.feedforward(x)),y) for (x,y) in data]
	 	return sum(int(x==y) for (x,y) in results)
	
	 def total_cost(self,data,lmbda,convert=False):
	 	cost = 0.0
	 	for x,y in data:
	 		a = self.feedforward(x)
	 		if convert: y = vectorized_label(y)
	 		cost += self.cost.costFunction(a,y)/len(data)
	 	cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

	 	return cost

	 def save(self,filename):

	 	data = {"sizes": self.sizes,
	 	         "weights": self.weights,
	 	         "biases": self.biases,
	 	         "cost": str(self.cost.__name__)}
	 	file = open(filename,"w")
	 	json.dump(data,file)
	 	file.close()


	 def backprop(self,x,y):

	 	gradient_bias = [np.zeros(b.shape) for b in self.biases]
	 	gradient_weight = [np.zeros(w.shape) for w in self.weights]
	 	activation = x
	 	activations = [x]
	 	z_values = []

	 	for b,w in zip(self.biases,self.weights):
	 		z = np.dot(w,activation)+b
	 		z_values.append(z)
	 		activation=sigmoid(z)
	 		activations.append(activation)

	 	delta = (self.cost).error_delta(z_values[-1],activations[-1],y)
	 	gradient_bias[-1]=delta
	 	gradient_weight[-1]=np.dot(delta,activations[-2].transpose())

		for i in xrange(2,self.num_of_layers):

			z = z_values[-i]
			delta = np.dot(self.weights[-i+1].transpose(),delta) * sigmoid_derivate(z)
			gradient_bias[-i] = delta
			gradient_weight[-i] = np.dot(delta,activations[-i-1].transpose())
		
		return gradient_bias,gradient_weight


def sigmoid_derivate(z):
	return sigmoid(z) * (1 - sigmoid(z))

def load(filename):
	file = open(filename,"r")
	data = json.load(file)
	file.close()
	cost = getattr(sys.modules[__name__],data["cost"])
	net = Network(data["sizes"],cost=cost)
	net.weights = [np.array(w) for w in data["weights"]]
	net.biases = [np.array(b) for b in data["biases"]]
	return net

def vectorized_label(label):
	vectorized_label = np.zeros((10,1))
	vectorized_label[label] = 1
	return vectorized_label

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))