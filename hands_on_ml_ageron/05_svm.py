import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC


def plot_estimator(estimator, X, y, axes=None):
    if axes is None:
        x0_std = np.std(X[:, 0])
        x1_std = np.std(X[:, 1])
        axes = [
            np.amin(X[:, 0]) - x0_std,
            np.amax(X[:, 0]) + x0_std,
            np.amin(X[:, 1]) - x1_std,
            np.amax(X[:, 1]) + x1_std,
        ]
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(211)
    plot_dataset(X, y, axes, ax1)
    plot_predictions(estimator, axes, ax1, decision=False)
    ax2 = fig.add_subplot(212)
    plot_dataset(X, y, axes, ax2)
    plot_predictions(estimator, axes, ax2, pred=False)


def plot_dataset(X, y, axes, ax=None):
    if ax is None:
        ax = plt
    ax.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs')
    ax.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'g^')
    ax.axis(axes)
    ax.grid(True, which='both')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$', rotation=0)


def plot_predictions(clf, axes, ax=None, pred=True, decision=True):
    if ax is None:
        ax = plt
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    if pred:
        ax.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    if decision:
        ax.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


def linear_svm():
    # from sklearn.linear_model import SGDClassifier

    iris = datasets.load_iris()
    X = iris['data'][:, [2, 3]]
    y = (iris['target'] == 2).astype(np.float64)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svm', LinearSVC(C=1, loss='hinge')),
        # ('linear_svm', SVC(C=1, kernel='linear')),
        # ('linear_svm', SGDClassifier(loss='hinge', alpha=1 / (X.shape[0] * 1), max_iter=5, tol=None)),
    ])

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    print('accuracy: {}'.format(np.mean(y == y_pred)))
    plot_estimator(pipeline, X, y)


def polynomial_svm():
    X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=66)
    real_poly = False
    if real_poly:
        pipeline = Pipeline([
            ('poly features', PolynomialFeatures(degree=3)),
            ('scale', StandardScaler()),
            ('clf', LinearSVC(C=5, loss='hinge')),
        ])
    else:
        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(C=5, kernel='poly', degree=3, coef0=1)),
        ])
    pipeline.fit(X, y)
    plot_estimator(pipeline, X, y)


def similarity_svm():
    X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=66)
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=100, gamma=5)),
    ])
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    print('accuracy: {}'.format(np.mean(y == y_pred)))
    plot_estimator(pipeline, X, y)


# linear_svm()
# polynomial_svm()
similarity_svm()
# import timeit
# print(timeit.timeit('linear_svm()', number=1000, setup='from __main__ import linear_svm'))
plt.show()
