import numpy as np
import nn

Xtr = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Ytr = np.array([[0], [1], [1], [0]]).T

ld = [2, 1]

lrate = 10

num_iterations = 10000

#print(Ytr.shape[1])


parameters = nn.model(X = Xtr, Y = Ytr, nn_architecture = ld, start_learning_rate = lrate, num_iterations = num_iterations, learning_decay = 0.01)

#parameters = nn.initialize_parameters(ld)

#print(nn.forward_propagation(Xtr, parameters)[0])

nn.predict([[0], [0]], parameters)
nn.predict([[1], [0]], parameters)
nn.predict([[0], [1]], parameters)
nn.predict([[1], [1]], parameters)