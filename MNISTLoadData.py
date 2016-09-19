import cPickle, gzip
import numpy as np

""" MNIST dataset is divided inot training,validation, and testing data.

This file returns training data,validation data and testing data as list
"""

def load_data():
	file = gzip.open('/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/mnist.pkl.gz','rb')
	training_set,validation_set,test_set = cPickle.load(file)
	file.close()
	return training_set,validation_set,test_set

def vectorized_labels(label):
	vectorized_label = np.zeros((10,1))
	vectorized_label[label] = 1
	return vectorized_label

def load_data_tuples():
	training_set,validation_set,test_set = load_data()
	training_images = [np.reshape(x,(784,1)) for x in training_set[0]]
	training_labels = [vectorized_labels(label) for label in training_set[1]]
	training_data = zip(training_images,training_labels)

	testing_images = [np.reshape(x,(784,1)) for x in test_set[0]]
	testing_data = zip(testing_images,test_set[1])

	validation_images = [np.reshape(x,(784,1)) for x in validation_set[0]]
	validation_data = zip(validation_images,validation_set[1])

	return (training_data,validation_data,testing_data)