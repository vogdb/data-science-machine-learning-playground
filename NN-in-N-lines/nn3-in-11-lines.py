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
y = np.array([[0, 1, 1, 0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# weights from the input layer to the hidden layer comprised of 5 nodes
layer1_weights = np.random.uniform(-1.0, 1.0, (3, 4))
# weights from hidden layer to output layer
layer2_weights = np.random.uniform(-1.0, 1.0, (4, 1))

# train
print('Training of two layer NN for input of shape {}'.format((X.shape[1], 1)))
print('Initial layer1 weights:\n{}'.format(layer1_weights))
print('Initial layer2 weights:\n{}'.format(layer2_weights))

for idx in range(50000):
    # forward propagation
    layer1 = nonlin(np.dot(X, layer1_weights))
    layer2 = nonlin(np.dot(layer1, layer2_weights))

    # how much did we miss?
    y_predict = layer2
    y_error = y - y_predict

    layer2_error = y_error
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    layer2_weights_delta = layer2_error * nonlin_derivative(layer2)
    # this multiplication is a part of the chained derivative
    layer2_weights_delta = np.dot(layer1.T, layer2_weights_delta)

    # the same logic as above
    layer1_error = np.dot(layer2_weights_delta.T, layer1)
    layer1_weights_delta = layer1_error * nonlin_derivative(layer1)
    layer1_weights_delta = np.dot(X.T, layer1_weights_delta)

    # update weights
    layer1_weights += layer1_weights_delta
    layer2_weights += layer2_weights_delta

    if idx % 5000 == 0:
        print('y_predict: {}, y_error: {}\nlayer1 weights:\n{}\nlayer2 weights:\n{}'.
              format(y_predict.flatten(), y_error.flatten(), layer1_weights, layer2_weights))
