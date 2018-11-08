import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap


def generate_dataset():
    iris = datasets.load_iris()
    X = iris['data'][:, [2, 3]]
    return X, iris['target']


def plot_countours(X, y, reg):
    ax = plt.gca()
    x0_min = 0
    x0_max = 8
    x1_min = 0
    x1_max = 3.5
    x0, x1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 500)[:, np.newaxis],
        np.linspace(x1_min, x1_max, 200)[:, np.newaxis],
    )

    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_proba = reg.predict_proba(X_new)
    y_predict = reg.predict(X_new)

    # plot samples on {x0, x1} grid
    ax.plot(X[y == 2, 0], X[y == 2, 1], 'g^', label='Iris-Virginica')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bs', label='Iris-Versicolor')
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'yo', label='Iris-Setosa')

    # plot countour plots of all classes
    y_predict_p = y_predict.reshape(x0.shape)
    contour = ax.contourf(x0, x1, y_predict_p, cmap=ListedColormap(['#fafab0', '#9898ff', '#a0faa0']))
    plt.clabel(contour, inline=1, fontsize=12)
    # plot probability countour plots of 1 class on {x0, x1} grid
    p1 = y_proba[:, 1].reshape(x0.shape)
    contour = ax.contour(x0, x1, p1, cmap=plt.cm.brg)
    ax.clabel(contour, inline=1, fontsize=12)

    ax.set_xlabel('Petal length', fontsize=14)
    ax.set_ylabel('Petal width', fontsize=14)
    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)
    plt.legend(loc='center left')

    plt.show()


class SoftmaxRegression():
    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.n_classes = len(set(y))

        self.X = self.bias_to_X(X)
        self.y = self.one_hot_y(y)
        self.theta = np.random.normal(size=(self.n_classes, self.X.shape[1]))
        self.gd()

    def bias_to_X(self, X):
        return np.hstack((
            np.ones((X.shape[0], 1)),
            X
        ))

    def one_hot_y(self, y):
        n_samples = y.shape[0]
        n_classes = len(set(y))

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1
        return y_one_hot

    def predict_proba(self, X):
        X = self.bias_to_X(X)
        return self.softmax(X)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def softmax(self, X):
        q = X.dot(self.theta)
        exps = np.exp(q)
        exps_sum = np.sum(exps, axis=1, keepdims=True)
        return exps / exps_sum

    def gd(self):
        epsilon = 1e-7
        lr = 0.01
        n_iterations = 5001

        def cost_function(p):
            return -np.mean(np.sum(self.y * np.log(p + epsilon), axis=1))

        for i in range(n_iterations):
            p = self.softmax(self.X)
            loss = cost_function(p)
            error = p - self.y
            theta_gradient = (1 / self.n_samples) * self.X.T.dot(error)
            self.theta -= lr * theta_gradient
            if i % 500 == 0:
                print(i, loss)


X, y = generate_dataset()
custom_softmax = True
if custom_softmax:
    test_ratio = 0.2
    total_size = X.shape[0]
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size
    rnd_indices = np.random.permutation(total_size)

    X_train = X[rnd_indices[:train_size]]
    y_train = y[rnd_indices[:train_size]]
    X_test = X[rnd_indices[-test_size:]]
    y_test = y[rnd_indices[-test_size:]]

    softmax_reg = SoftmaxRegression()
    softmax_reg.fit(X_train, y_train)
    y_test_predict = softmax_reg.predict(X_test)
    print('Custom softmax accuracy: {}'.format(np.mean(y_test == y_test_predict)))

else:
    softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10, random_state=42)
    softmax_reg.fit(X, y)
    plot_countours(X, y, softmax_reg)
