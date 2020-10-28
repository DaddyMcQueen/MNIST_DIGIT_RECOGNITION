import numpy as np
import pickle
import pygame
from nn import DeepNN
from mnistload import Batch
from predict_drawing import get_drawing
from deepnn.constants import SIZE_IN
from deepnn.constants import SIZE_OUT
from deepnn.constants import SIZE_HIDDEN
from deepnn.constants import LEARNING_RATE
from deepnn.constants import NUM_BATCHES
from deepnn.constants import EPOCHS

#-----------------------#
# MNIST Dataset problem
# Deep Neural Network
#
# Version: 10-28-2020
#-----------------------#
def main():
	new_network = False
	test_network = True
	predict_drawing = False
	
	if new_network:
		mnist_network = DeepNN(SIZE_IN, SIZE_OUT, SIZE_HIDDEN, LEARNING_RATE)
		train_network = True
	else:
		pickel_in = open("nn.pickel", "rb")
		mnist_network = pickle.load(pickel_in)
		train_network = False

	while predict_drawing:
		drawing, run = get_drawing()
		if run == False:
			predict_drawing = False
			break
		outputs = mnist_network.forward_pass(drawing).T
		prediction_val = outputs[0]
		prediction = 0
		for j in range(9):
			if outputs[j] > prediction_val:
				prediction_val = outputs[j]
				prediction = j

		print(prediction)
		
	if train_network == True:
		print("Training Network:")
		print("   Epoch   | Accuracy ")

		t_history = []
		for i in range(EPOCHS):
			accuracy = 0
			for K in range(20):
				test_batch = Batch(K + 579)
				test_images = test_batch.images
				test_labels = test_batch.labels
				accuracy += mnist_network.test_network(test_images, test_labels)
			avg_accuracy = accuracy * 5
			t_history.append(avg_accuracy)

			if i <10:
				print("    ", i, "    |  ", round(avg_accuracy, 2) ,"%  ")
			elif i >100:
				print("    ", i, "   |  ", round(avg_accuracy, 2) ,"%  ")
			else:
				print("    ", i, "   |  ", round(avg_accuracy, 2) ,"%  ")
			for j in range(579):
				batch = Batch(j)
				inputs = batch.images
				targets = batch.labels
				mnist_network.train(inputs, targets)
		pickle_out = open("nn.pickel", "wb")
		pickle.dump(mnist_network, pickle_out)
		pickle_out.close()
				
	# get testing data 
	if test_network == True:
		accuracy = 0
		for i in range(20):
			test_batch = Batch(i + 579)
			test_images = test_batch.images
			test_labels = test_batch.labels
			accuracy += mnist_network.test_network(test_images, test_labels)
		avg_accuracy = accuracy/20
		print("The accuracy of the network on never before seen data is: ", avg_accuracy * 100, "%")
main()
