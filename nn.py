import numpy as np
from time import time

np.random.seed(1)#int(time()))

def sigmoid(X):
	return (1 / (1 + (1 / np.exp(X))))

def tanh(Z):
	return np.divide((np.exp(Z) - np.exp(-Z)), (np.exp(Z) + np.exp(-Z)))

def initialize_parameters(layer_dims):
	parameters = {}
	L = len(layer_dims)

	for l in range(1, L):
		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])# * 0.1
		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
	return parameters

def forward_propagation(X, parameters):
	caches = []
	caches.append((0, X))
	L = len(parameters) // 2
	A = X

	for l in range(1, L):
		Z = np.dot(parameters["W" + str(l)], A) + parameters["b" + str(l)]
		A = tanh(Z)
		caches.append((Z, A))

	# sigmoid activation on output layer
	Z = np.dot(parameters["W" + str(L)], A) + parameters["b" + str(L)]
	A = sigmoid(Z)
	caches.append((Z, A))

	return A, caches

def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = -np.sum( np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)) ) / m

	return cost

def backward_propagation(AL, Y, caches, parameters):
	grads = {}
	
	L = len(parameters) // 2
	m = Y.shape[1]

	#dAL =  -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
	#grads["dA" + str(L)] = dAL
 
 	#Derivatives for output (sigmoid) layer
	grads["dZ" + str(L)] = (AL - Y) / m
	grads["dW" + str(L)] = np.dot(grads["dZ" + str(L)], caches[L - 1][1].T)
	grads["db" + str(L)] = np.sum(grads["dZ" + str(L)], axis = 1, keepdims=True)

	for l in reversed(range(1, L)):
		grads["dA" + str(l)] = np.dot(parameters["W" + str(l+1)].T, grads["dZ" + str(l+1)])
		grads["dZ" + str(l)] = np.multiply((1 - caches[l][1] ** 2), grads["dA" + str(l)]) / m
		grads["dW" + str(l)] = np.dot(grads["dZ" + str(l)], caches[l - 1][1].T)
		grads["db" + str(l)] = np.sum(grads["dZ" + str(l)], axis = 1, keepdims=True)

	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2

	for l in range(1, L+1):
		parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
		parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

	return parameters

def model(X, Y, nn_architecture, start_learning_rate = 0.0075, num_iterations = 1500, print_cost = True, learning_decay = 0.01):

	layer_dims = [X.shape[0]]
	for i in nn_architecture:
		layer_dims.append(i)

	parameters = initialize_parameters(layer_dims)

	for i in range(0, num_iterations):
		AL, caches = forward_propagation(X, parameters)
		cost = compute_cost(AL, Y)
		grads = backward_propagation(AL, Y, caches, parameters)
		learning_rate = start_learning_rate * (1 / (1 + learning_decay * i))
		parameters = update_parameters(parameters, grads, learning_rate)

		if print_cost and i % 100 == 0:
			print("Cost after iteration %i: %f" %(i, cost))
			#costs.append(cost)

		if cost < 0.000001:
			print("Cost after iteration %i: %f" %(i, cost))
			break

	return parameters

def predict(X, parameters):
	Y, _ = forward_propagation(X, parameters)
	Y = int(Y[0][0] > 0.5)
	print ("Prediction is", Y)