import numpy as np
from deepnn.constants import SIZE_IN
from deepnn.constants import SIZE_OUT
from deepnn.constants import SIZE_HIDDEN
from deepnn.constants import LEARNING_RATE
from deepnn.constants import NUM_BATCHES
from nn import DeepNN
from mnistload import Batch

#-----------------------#
# MNIST Dataset problem
# Deep Neural Network
#
# Author: Jacob Giczi 
# Version: 10-20-2020
#-----------------------#

def create_network():
	mnist_network = DeepNN(SIZE_IN, SIZE_OUT, SIZE_HIDDEN, LEARNING_RATE)

def main():
	create_network = True

	if create_network == True:
		mnist_network = DeepNN(SIZE_IN, SIZE_OUT, SIZE_HIDDEN, LEARNING_RATE)


	train_network = True
	
	if train_network == True:
		for i in range(598):
			batch = Batch(i)
			inputs = batch.batch_i
			targets = batch.batch_l
			mnist_network.train(inputs, targets)
			if i >= 599:
				pass
				# print(mnist_network.w_in_hidden1)
				# print(mnist_network.w_hidden1_hidden2)
				# print(mnist_network.w_hidden2_out)
				# print(mnist_network.bias_h1)
				# print(mnist_network.bias_h2)
				# print(mnist_network.bias_out)

	test_network = True

	if test_network == True:
		test_batch = Batch(599)
		test_images = test_batch.batch_i
		test_labels = test_batch.batch_l
		print(test_labels[17])
		mnist_network.forward_pass(test_images[17])
		print(mnist_network.layer_out)
main()