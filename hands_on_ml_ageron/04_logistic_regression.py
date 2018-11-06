from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


def generate_dataset():
    iris = datasets.load_iris()
    # petal length, petal width
    X = iris['data'][:, [2, 3]]
    # 1 if Iris-Virginica, else 0
    y = (iris['target'] == 2).astype(np.int)
    return X, y


def plot_countours(X, y):
    ax = plt.gca()
    x0_min = 2.9
    x0_max = 7
    x1_min = 0.8
    x1_max = 2.7
    x0, x1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 500)[:, np.newaxis],
        np.linspace(x1_min, x1_max, 200)[:, np.newaxis],
    )

    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_proba = log_reg.predict_proba(X_new)

    # plot samples on {x0, x1} grid
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'bs')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'g^')

    # plot probability countour plots of 1 class on {x0, x1} grid
    p1 = y_proba[:, 1].reshape(x0.shape)
    contour = ax.contour(x0, x1, p1, cmap=plt.cm.brg)
    ax.clabel(contour, inline=1, fontsize=12)

    # plot the learned params
    left_right = np.array([x0_min, x0_max])
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
    ax.plot(left_right, boundary, 'k--', linewidth=3)

    ax.text(3.5, 1.5, 'Not Iris-Virginica', fontsize=14, color='b', ha='center')
    ax.text(6.5, 2.3, 'Iris-Virginica', fontsize=14, color='g', ha='center')
    ax.set_xlabel('Petal length', fontsize=14)
    ax.set_ylabel('Petal width', fontsize=14)
    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)

    plt.show()


def cost_function(X, y):
    m = y.shape[0]
    theta_min = -10
    theta_max = 10

    theta = np.arange(theta_min, theta_max, 0.5)
    theta1, theta2 = np.meshgrid(theta, theta)
    theta = np.hstack((theta[:, np.newaxis], theta[:, np.newaxis]))
    c = np.zeros(theta1.shape)

    for i in range(m):
        p = expit(X[i].dot(theta.T))
        c += -(np.log(p) * y[i] + np.log(1 - p) * (1 - y[i]))
    c = c / m

    ax = plt.gca(projection='3d')
    ax.plot_surface(theta1, theta2, c)
    plt.show()


X, y = generate_dataset()
log_reg = LogisticRegression(C=10 ** 10, random_state=42)
log_reg.fit(X, y)

# cost_function(X, y)
plot_countours(X, y)
