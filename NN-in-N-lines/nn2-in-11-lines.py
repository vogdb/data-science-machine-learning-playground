import numpy as np


# sigmoid function
def nonlin(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative with respect to weights
def nonlin_derivative(x):
    return x * (1 - x)


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# Network consists of single neuron with 3 weights and 0 biases
network_layer_weights = np.random.uniform(-1.0, 1.0, (3, 1))

# add bias to weights
network_layer_bias = np.random.uniform(-1.0, 1.0, (1, 1))
network_layer_weights = np.vstack((network_layer_weights, network_layer_bias))
# inject a column of ones into X for bias multiplication
X = np.hstack((X, np.ones((X.shape[0], 1))))

# train
print('Training of a single neuron NN for input of shape {}'.format((X.shape[1], 1)))
print('Initial weights: {}'.format(network_layer_weights.flatten()))
for idx in range(100):
    # forward propagation
    network_input = X
    y_predict = nonlin(np.dot(network_input, network_layer_weights))

    # how much did we miss?
    y_error = y - y_predict

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    weights_delta = y_error * nonlin_derivative(y_predict)

    # update weights
    network_layer_weights += np.dot(network_input.T, weights_delta)
    print('y_predict: {}, y_error: {}, weights: {}'.
          format(y_predict.flatten(), y_error.flatten(), network_layer_weights.flatten()))
