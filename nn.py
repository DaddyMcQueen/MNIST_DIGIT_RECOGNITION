import numpy as np
from deepnn.constants import WEIGHT_MULTIPLIER
from deepnn.constants import SIZE_OUT
from deepnn.constants import LEARNING_RATE
from deepnn.constants import BATCH_SIZE

class DeepNN:

	def __init__(self, n_in_neurons, n_out_neurons, n_hidden_neutrons, learning_rate):

		self.n_in_neurons = n_in_neurons
		self.n_out_neurons = n_out_neurons
		self.n_hidden_neutrons = n_hidden_neutrons
		self.learning_rate = learning_rate

		self.goals = []

		#Creating desired arrays of random weights, zeros, and goals
		self.create_weights()
		self.create_bias()
		self.create_goals()
		self.reset_deltas()
				
	# generating random weights for all neurons in network
	def create_weights(self):
		np.random.seed(0)

		#weight arrays for input to first hidden layer
		self.w_in_hidden1 = WEIGHT_MULTIPLIER * np.random.randn(self.n_in_neurons, self.n_hidden_neutrons)

		#weight arrays for hidden layers
		self.w_hidden1_hidden2 = WEIGHT_MULTIPLIER * np.random.randn(self.n_hidden_neutrons, self.n_hidden_neutrons)
		
		#weight array for final layer
		self.w_hidden2_out = WEIGHT_MULTIPLIER * np.random.randn(self.n_hidden_neutrons, self.n_out_neurons)

	#generating 0's arrays for biases
	def create_bias(self):
		#bias matrix for first hidden layer
		self.bias_h1 = np.zeros((1, self.n_hidden_neutrons))

		#bias matrix for second layer
		self.bias_h2 = np.zeros((1, self.n_hidden_neutrons))

		#bias matrix for output later
		self.bias_out = np.zeros((1, self.n_out_neurons))

	#make deltas 0
	def reset_deltas(self):
		self.w_hidden2_out_delta = np.zeros((self.n_hidden_neutrons, SIZE_OUT))
		self.w_hidden1_hidden2_delta = np.zeros((self.n_hidden_neutrons, self.n_hidden_neutrons))
		self.w_in_hidden1_delta = np.zeros((self.n_in_neurons, self.n_hidden_neutrons))
		self.bias_out_delta = np.zeros((1, self.n_out_neurons))
		self.bias_h2_delta = np.zeros((1, self.n_hidden_neutrons))
		self.bias_h1_delta = np.zeros((1, self.n_hidden_neutrons))

	#list of 1 d arrays with goal values for calculating cost
	def create_goals(self):
		for i in range(SIZE_OUT):
			goal = np.zeros((SIZE_OUT, 1))
			goal[i] = 1
			self.goals.append(goal)

	#squish numbers between -1 and 1
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def d_sigmoid(self, x):
		return x * (1 - x)

	#preform forward pass through NN self.layer_out is 1 by 10 array 0 <= values <= 1
	def forward_pass(self, inputs):
		inputs = np.asarray(inputs) 

		self.layer_0  = inputs.reshape(1, 784)
		self.layer_h1 = self.sigmoid(np.dot(self.layer_0, self.w_in_hidden1) + self.bias_h1)
		self.layer_h2 = self.sigmoid(np.dot(self.layer_h1, self.w_hidden1_hidden2) + self.bias_h2)
		self.layer_out = self.sigmoid(np.dot(self.layer_h2, self.w_hidden2_out) + self.bias_out)

		return self.layer_out

	#cost of each training sample calculated as twice the difference between output and corisponding goal array
	def costs(self, outputs, goals):
		costs = (outputs.T - goals) * 2
		return costs

	#proform back prop based on costs
	#"I just kept adding more lin alg untill it worked" - Jacob Giczi
	def back_prop(self):
		self.w_hidden2_out_delta += (self.output_costs * self.layer_h2).T * self.d_sigmoid(self.layer_out)  
		self.bias_out_delta += (self.d_sigmoid(self.layer_out) * self.output_costs.T)

		w_hidden2_costs = self.w_hidden2_out.dot(self.output_costs) * self.d_sigmoid(self.layer_out)
		w_hidden2_costs = np.sum(w_hidden2_costs, axis=1)				
		
		self.w_hidden1_hidden2_delta += (w_hidden2_costs * self.layer_h1).T * self.d_sigmoid(self.layer_h2)
		self.bias_h2_delta += (self.d_sigmoid(self.layer_h2) * w_hidden2_costs.T)

		w_hidden1_costs = (self.w_hidden1_hidden2 * w_hidden2_costs) * self.d_sigmoid(self.layer_h2)
		w_hidden1_costs = np.sum(w_hidden1_costs, axis=1)

		self.w_in_hidden1_delta += self.layer_0.T * w_hidden1_costs * self.d_sigmoid(self.layer_h1)
		self.bias_h1_delta += (self.d_sigmoid(self.layer_h1) * w_hidden1_costs.T)

	#update weights and biases by the average delta of each batch
	def update_w_b(self):
		#update weights
		self.w_in_hidden1 -= self.w_in_hidden1_delta * LEARNING_RATE / BATCH_SIZE
		self.w_hidden1_hidden2 -= self.w_hidden1_hidden2_delta * LEARNING_RATE / BATCH_SIZE
		self.w_hidden2_out -= self.w_hidden2_out_delta * LEARNING_RATE / BATCH_SIZE

		#update biases
		self.bias_h1 -= self.bias_h1_delta * LEARNING_RATE / BATCH_SIZE
		self.bias_h2 -= self.bias_h2_delta * LEARNING_RATE / BATCH_SIZE
		self.bias_out -= self.bias_out_delta * LEARNING_RATE / BATCH_SIZE
		self.reset_deltas()

	#preform forward pass, followed by back prop for a single batch
	def train(self, inputs, targets):

		#split data into the image and target number
		self.targets = targets
		self.inputs = inputs

		#take batchsize number of inputs
		for i in range(BATCH_SIZE):
			outputs = self.forward_pass(self.inputs[i])
			self.output_costs = self.costs(outputs, self.goals[self.targets[i]])
			self.back_prop()

		self.update_w_b()

	def test_network(self, inputs, labels):
		correct = 0
		for i in range(BATCH_SIZE):
			outputs = self.forward_pass(inputs[i]).T
			print(outputs)
			print(labels[i])
			guess_val = outputs[0]
			guess = 0
			for j in range(10):
				if outputs[j] > guess_val:
					guess_val = outputs[j]
					guess = j
			print(guess)
			if guess == labels[i]:
				correct += 1
		return correct / BATCH_SIZE
