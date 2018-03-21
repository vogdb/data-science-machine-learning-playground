import numpy as np
p = 0.5 # probability of keeping a unit active. higher = less dropout

X = np.random.rand(2, 4)

W1 = np.random.rand(3, 4)
b1 = np.random.rand(3, 1)
W2 = np.random.rand(2, 3)
b2 = np.random.rand(2, 1)
W3 = np.random.rand(1, 2)
b3 = np.random.rand(1, 1)

print 'W1\n', W1
print 'b1\n', b1
print 'W2\n', W2
print 'b2\n', b2
print 'W3\n', W3
print 'b3\n', b3

def train_step(X):
    """ X contains the data """

    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X.T) + b1)
    print 'H1\n', H1
    U1 = np.random.rand(*H1.shape) < p # first dropout mask
    print 'U1\n', U1
    H1 *= U1 # drop!
    print 'H1 after drop\n', H1
    H2 = np.maximum(0, np.dot(W2.T, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p # second dropout mask
    H2 *= U2 # drop!
    out = np.dot(W3.T, H2) + b3


train_step(X)