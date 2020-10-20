from mnist import MNIST
import numpy as np

from deepnn.constants import BATCH_SIZE
from deepnn.constants import PIXELS

mndata = MNIST('data/mnist/')
images, labels = mndata.load_training()

batches = len(labels) / BATCH_SIZE

all_images = np.asarray(images)
all_images = all_images.reshape(int(batches), BATCH_SIZE, PIXELS)
all_images = all_images / 255
# print(np.shape(all_images))

all_labels = np.asarray(labels)
all_labels = all_labels.reshape(int(batches), BATCH_SIZE)
# print(np.shape(all_labels))



class Batch:
	def __init__(self, batch_num):
		self.images = all_images[batch_num]
		self.labels = all_labels[batch_num]
		self.used_images = 0
		self.create_batch(batch_num)


	def create_batch(self, batch_num):
		self.batch_i = self.images
		self.batch_l = self.labels
		