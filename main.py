import numpy as np
import pickle
from deepnn.constants import SIZE_IN
from deepnn.constants import SIZE_OUT
from deepnn.constants import SIZE_HIDDEN
from deepnn.constants import LEARNING_RATE
from deepnn.constants import NUM_BATCHES
from deepnn.constants import EPOCHS
from nn import DeepNN
from mnistload import Batch

#-----------------------#
# MNIST Dataset problem
# Deep Neural Network
#
# Version: 10-23-2020
#-----------------------#
def main():
	new_network = True
	test_network = True

	if new_network:
		mnist_network = DeepNN(SIZE_IN, SIZE_OUT, SIZE_HIDDEN, LEARNING_RATE)
		train_network = True
	else:
		pickel_in = open("nn.pickel", "rb")
		mnist_network = pickle.load(pickel_in)
		train_network = False
		
	if train_network == True:

		for i in range(EPOCHS):
			for i in range(549):
				batch = Batch(i)
				inputs = batch.images
				targets = batch.labels
				mnist_network.train(inputs, targets)
		pickle_out = open("nn.pickel", "wb")
		pickle.dump(mnist_network, pickle_out)
		pickle_out.close()
				
	# get testing data 
	if test_network == True:
		accuracy = 0
		for i in range(50):
			test_batch = Batch(i + 549)
			test_images = test_batch.images
			test_labels = test_batch.labels
			accuracy += mnist_network.test_network(test_images, test_labels)
		avg_accuracy = accuracy/50
		print("The accuracy of the network on never before seen data is: ", avg_accuracy * 100, "%")
main()
