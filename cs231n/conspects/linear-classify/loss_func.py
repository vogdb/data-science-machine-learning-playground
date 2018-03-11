import numpy as np


def L_i(x, y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
      with an appended bias dimension in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
    """
    delta = 1.0  # see notes about delta later in this section
    scores = W.dot(x)  # scores becomes of size 10 x 1, the scores for each class
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes, e.g. 10
    loss_i = 0.0
    for j in xrange(D):  # iterate over all wrong classes
        if j == y:
            # skip for the true class to only loop over incorrect classes
            continue
        # accumulate loss for the i-th example
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    scores = W.dot(x)
    print 'scores', scores
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def L(X, y, W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    delta = 1.0
    scores = W.dot(X.T)
    expected_score = scores[y, range(scores.shape[1])]
    margins = np.maximum(0, scores - expected_score + delta)
    loss = np.sum(margins) - delta * X.shape[0]
    return loss


# a = np.ones((3, 3))
# print a
# b = np.array([1, 2, 3])
# b = np.array([[1, 2, 3]])
# print b
#
# print a - b
#
# # print b.reshape(3, 1)
# print b.T
# print a - b.T


np.random.seed(111)

W = np.random.rand(3, 5)

X = np.random.randint(low=1, high=50, size=(4, 5))
X[:, X.shape[1] - 1] = 1

y = np.random.randint(low=0, high=W.shape[0], size=X.shape[0])

print "W", W
print "X", X
print "y", y

L_vectorized = 0
for idx, x in enumerate(X):
    L_i = L_i_vectorized(x, y[idx], W)
    L_vectorized += L_i

print 'Vectorized L', L_vectorized

print 'Matrix L', L(X, y, W)
