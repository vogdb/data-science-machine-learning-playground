import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def generate_dataset():
    size = 100
    X = 2 * np.random.random(size=(size, 1))
    y = 4 * X + 3 + np.random.randn(size, 1)
    return X, y


def sklearn_regression(X, y):
    sk_linreg = LinearRegression()
    sk_linreg.fit(X, y)
    print('sklearn params: {}, {}'.format(sk_linreg.intercept_, sk_linreg.coef_))


def grad_descent_regression(X, y):
    theta = np.random.rand(2, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n_iter = 1000
    lr = 0.01
    common_factor = 2 / X.shape[0]
    for i in range(n_iter):
        y_pred = X.dot(theta)
        # X.T is to make: x_j * (y_pred - y) for x_j of all X samples
        mse_diff_theta = common_factor * X.T.dot(y_pred - y)
        d_theta = -mse_diff_theta
        theta = theta + lr * d_theta
    print('grad descent params: {}, {}'.format(theta[0], theta[1]))


def mini_batch_grad_descent_regression(X, y):
    theta = np.random.rand(2, 1)
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    n_epoch = 200
    mini_batch_size = 5
    lr = 0.01
    common_factor = 2 / m
    epoch_size = int(m / mini_batch_size)

    for i in range(n_epoch):
        for j in range(epoch_size):
            rnd_idx = np.random.randint(0, m, size=mini_batch_size)
            X_mbatch = X[rnd_idx]
            y_pred = X_mbatch.dot(theta)
            mse_diff_theta = common_factor * X_mbatch.T.dot(y_pred - y[rnd_idx])
            d_theta = -mse_diff_theta
            theta = theta + lr * d_theta
    print('mini batch grad descent params: {}, {}'.format(theta[0], theta[1]))


X, y = generate_dataset()
plt.plot(X, y, '.')
# plt.show()

sklearn_regression(X, y)
grad_descent_regression(X, y)
mini_batch_grad_descent_regression(X, y)
