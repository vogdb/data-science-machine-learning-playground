import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(66)


def generate_dataset(size):
    X = 6 * np.random.rand(size, 1) - 3
    y = .5 * X ** 2 + X + 2 + np.random.randn(size, 1)
    return X, y


def plot_data_predictions(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    lin_reg = LinearRegression()
    poly2_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly2 = poly2_features.fit_transform(X)
    lin_reg.fit(X_poly2, y)
    y_pred_poly2 = lin_reg.predict(X_poly2)

    lin_reg = LinearRegression()
    poly10_features = PolynomialFeatures(degree=20, include_bias=False)
    X_poly10 = poly10_features.fit_transform(X)
    lin_reg.fit(X_poly10, y)
    y_pred_poly10 = lin_reg.predict(X_poly10)

    plt.plot(X, y, 'b.', label='data')
    plt.plot(X, y_pred_lin, 'r.', label='linear')
    plt.plot(X, y_pred_poly2, 'g.', label='poly 2')
    plt.plot(X, y_pred_poly10, 'y.', label='poly 10')
    plt.legend()
    plt.show()


def plot_model_learning_curves(model, color, name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    m = len(X_train)
    train_errors = []
    test_errors = []
    for i in range(1, m):
        model.fit(X_train[:i], y_train[:i])
        y_train_pred = model.predict(X_train[:i])
        train_errors.append(mean_squared_error(y_train[:i], y_train_pred))
        y_test_pred = model.predict(X_test)
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    plt.plot(range(1, m), train_errors, '-', color=color, label='train err ' + name)
    plt.plot(range(1, m), test_errors, '-.', color=color, label='val err ' + name)


def plot_learning_curves(X, y):
    lin_reg = LinearRegression()
    plot_model_learning_curves(lin_reg, 'g', 'lin', X, y)

    lin_reg = LinearRegression()
    poly2_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly2 = poly2_features.fit_transform(X)

    plot_model_learning_curves(lin_reg, 'b', 'poly2', X_poly2, y)

    plt.legend()
    plt.show()


m = 100
X, y = generate_dataset(m)
plot_learning_curves(X, y)
