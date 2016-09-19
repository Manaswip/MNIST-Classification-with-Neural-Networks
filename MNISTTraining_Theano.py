import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv 
from theano.tensor.nnet import softmax
from theano.tensor.signal import pool 
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import sigmoid
import cPickle, gzip

""" 
Code supports different kind of layer types( Full connected layer, convoluations layer, 
max pooling layer, softmax layer) and different activation functions (sigmoid, rectified linear
units, tanh)

Code is built using Theano library so, thiis code will be able to run either on CPU or GPU 
set GPU to true to run on GPU and set GPU to false to run on CPU

This program incorparates ideas and code from text book on 'Neural Networks and Deep learning' 
and github https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network3.py
from Michael Nielsen 
"""

GPU = True

if GPU:
	print "Running on GPU. To run on CPU set variable GPU to False"
	try: theano.config.floatX = 'float32'
else:
	print "Running on CPU. To run on GPU set variable GPU to True"

def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)

def load_data_shared():
	file = gzip.open('/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/mnist.pkl.gz','rb')
	training_set,validation_set,test_set = cPickle.load(file)
	file.close()
	def shared_variable(data):
		# Using shared variables data is loaded into GPU memory all at once.
		data_images,data_labels = data
		#data should be stored as floats on GPU, even though labels is integers store as floats and then cast
		shared_images = theano.shared(np.asarray(data_images,dtype=theano.config.floatX))
		shared_labels = theano.shared(np.asarray(data_labels,dtype=theano.config.floatX))
		return shared_images, T.cast(shared_labels,'int32')
	return [shared_variable(training_set),shared_variable(validation_set),shared_variable(test_set)]
	

class FullyConnectedLayer(object):

	def __init__(self,n_in,n_out,activation_fn=sigmoid,p_dropout=0.0):

		self.n_in = n_in
		self.n_out = n_out
		self.p_dropout = p_dropout
		self.activation_fn = activation_fn
		self.w = theano.shared(np.asarray(np.random.normal(loc=0.0,scale=np.sqrt(1.0/n_out),size=(n_in,n_out)),
			dtype=theano.config.floatX),name='w',borrow=True)
		self.b= theano.shared(np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(n_out,)),
			dtype=theano.config.floatX),name='b',borrow=True)
		self.params = [self.w, self.b]

	def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
		self.inpt = inpt.reshape((mini_batch_size,self.n_in))
		self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size,self.n_in)),self.p_dropout)
		self.output = self.activation_fn((1-self.p_dropout)*T.dot(self.inpt,self.w)+self.b)
		self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout,self.w)+self.b)
		self.y_out = T.argmax(self.output,axis=1)

	def accuracy(self,y):
		return T.mean(T.eq(y,self.y_out))

class ConvPoolLayer(object):

	def __init__(self,filter_shape,image_shape,poolsize=(2,2),activation_fn=sigmoid):

		self.filter_shape=filter_shape
		self.image_shape = image_shape
		self.activation_fn = activation_fn
		self.poolsize = poolsize
		n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
		self.w = theano.shared(np.asarray(np.random.normal(loc=0.0,scale=np.sqrt(1.0/n_out),size=filter_shape),
			dtype=theano.config.floatX),name='w',borrow=True)
		self.b = theano.shared(np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(filter_shape[0],)),
			dtype=theano.config.floatX),name='b',borrow=True)
		self.params = [self.w,self.b]

	def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
		self.inpt = inpt.reshape(self.image_shape)
		conv_out = conv.conv2d(
			input = self.inpt,filters=self.w,filter_shape=self.filter_shape,
			image_shape=self.image_shape)
		pooled_out = pool.pool_2d(
			input=conv_out,ds=self.poolsize,ignore_border=True)
		self.output = self.activation_fn(pooled_out+self.b.dimshuffle('x',0,'x','x'))
		self.output_dropout = self.output

class SoftmaxLayer(object):

	def __init__(self,n_in,n_out,p_dropout=0.0):
		self.n_in=n_in
		self.n_out=n_out
		self.w = theano.shared(np.zeros((n_in,n_out),dtype=theano.config.floatX),name='w',borrow=True)
		self.b = theano.shared(np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
		self.params = [self.w,self.b]
		self.p_dropout=p_dropout

	def set_inpt(self,inpt,inpt_dropout,mini_batch_size):
		self.inpt = inpt.reshape((mini_batch_size,self.n_in))
		self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size,self.n_in)),self.p_dropout)
		self.output = softmax((1-self.p_dropout)*T.dot(self.inpt,self.w)+self.b)
		self.y_out = T.argmax(self.output,axis=1)
		self.output_dropout = softmax(T.dot(self.inpt,self.w)+self.b)

	def cost(self,net):
		return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]),net.y])

	def accuracy(self,y):
		return T.mean(T.eq(y,self.y_out))

def size(data):
	return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer,p_dropout):
	srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
	mask = srng.binomial(n=1,p=1- p_dropout,size=layer.shape)
	return layer*T.cast(mask,theano.config.floatX)


class Network(object):

	def __init__(self,layers,mini_batch_size):
		self.layers=layers
		self.mini_batch_size=mini_batch_size
		self.params = [param for layer in self.layers for param in layer.params]
		self.x = T.matrix("x")
		self.y = T.ivector("y")
		init_layer = self.layers[0]
		init_layer.set_inpt(self.x,self.x,self.mini_batch_size)
		for j in xrange(1,len(self.layers)):
			prev_layer,layer = self.layers[j-1],self.layers[j]
			layer.set_inpt(prev_layer.output,prev_layer.output_dropout,self.mini_batch_size)
		self.output = self.layers[-1].output
		self.output_dropout = self.layers[-1].output_dropout

	def SGD(self,training_data,epochs,mini_batch_size,eta,validation_data,test_data,lmbda=0.0):
		training_x,training_y = training_data
		validation_x,validation_y=validation_data
		test_x,test_y=test_data	

		num_training_batches = size(training_data)/mini_batch_size
		num_validation_batches = size(validation_data)/mini_batch_size
		num_test_batches = size(test_data)/mini_batch_size

		l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
		cost = self.layers[-1].cost(self) +\
		 	   0.5*lmbda*l2_norm_squared/num_training_batches
		grads = T.grad(cost,self.params)
		updates = [(param,param-eta*grad) for param,grad in zip(self.params,grads)]
		i = T.lscalar()
		train_mb = theano.function(
		 	[i],cost, updates=updates,
		 	givens = {
		 	          self.x : training_x[i*mini_batch_size: (i+1)*mini_batch_size],
		 			  self.y: training_y[i*mini_batch_size: (i+1)*mini_batch_size]
		 			  }
		)

		validate_mb_accuracy = theano.function(
		 	[i],self.layers[-1].accuracy(self.y),
		 	givens = {
		 	          self.x : validation_x[i*mini_batch_size: (i+1)*mini_batch_size],
		 			  self.y: validation_y[i*mini_batch_size: (i+1)*mini_batch_size]		 		
		 	         }
		)
		test_mb_accuracy = theano.function(
		 	[i],self.layers[-1].accuracy(self.y),
		 	givens = {
		 	          self.x : test_x[i*mini_batch_size: (i+1)*mini_batch_size],
		 			  self.y: test_y[i*mini_batch_size: (i+1)*mini_batch_size]		 		
		 	         }
		)
		self.test_mb_predictions = theano.function(
		 	[i],self.layers[-1].y_out,
		 	givens = {
		 	          self.x : test_x[i*mini_batch_size: (i+1)*mini_batch_size]	 		
		 	         }
		)
		best_validation_accuracy = 0.0
		for epoch in xrange(epochs):
		 	for minibatch_index in xrange(num_training_batches):
		 		iteration = num_training_batches*epoch+minibatch_index
		 		if iteration%1000 == 0:
		 			print("training mini-batch number {0}".format(iteration))
		 		cost_ij = train_mb(minibatch_index)
		 		if (iteration+1)%num_training_batches ==0:
		 			validation_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
		 			print ("epoch: {0} , validation accuracy: {1}".format(epoch,validation_accuracy))
		 			if validation_accuracy >= best_validation_accuracy:
		 				print("This is the best accuracy up to date")
		 				best_validation_accuracy= validation_accuracy
		 				best_iteration = iteration
		 				if test_data:
		 					test_accuracy = np.mean([test_mb_accuracy(i) for i in xrange(num_test_batches)])
		 					print("the corresponding test accuracy is {0}".format(test_accuracy))




