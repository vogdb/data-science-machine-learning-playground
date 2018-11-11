import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import reciprocal, uniform
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR


def exercise8():
    def get_w_b_in_x1x2(svm):
        # w1 * x1 + w2 * x2 + b = 0 => x2 = -w1/w2 * x1 - b/w2
        w = -svm.coef_[0, 0] / svm.coef_[0, 1]
        b = -svm.intercept_[0] / svm.coef_[0, 1]
        return w, b

    X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, center_box=(0, 10), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    linear_svc = LinearSVC(C=1, loss='hinge')
    svc = SVC(C=1, kernel='linear')
    linear_sgd = SGDClassifier(loss='hinge', alpha=1 / (X.shape[0] * 1), max_iter=10, tol=None)

    linear_svc.fit(X_scaled, y)
    svc.fit(X_scaled, y)
    linear_sgd.fit(X_scaled, y)

    print('LinearSVC, b: {}, w: {}'.format(linear_svc.intercept_, linear_svc.coef_))
    print('SVC, b: {}, w: {}'.format(svc.intercept_, svc.coef_))
    print('LinearSGD, b: {}, w: {}'.format(linear_sgd.intercept_, linear_sgd.coef_))

    plt.plot(X_scaled[:, 0][y == 0], X_scaled[:, 1][y == 0], 'g^')
    plt.plot(X_scaled[:, 0][y == 1], X_scaled[:, 1][y == 1], 'bs')

    x_lim = np.array([np.amin(X_scaled), np.amax(X_scaled)])
    w1, b1 = get_w_b_in_x1x2(linear_svc)
    w2, b2 = get_w_b_in_x1x2(svc)
    w3, b3 = get_w_b_in_x1x2(linear_sgd)

    plt.plot(x_lim, x_lim * w1 + b1, 'r:', label='LinearSVC')
    plt.plot(x_lim, x_lim * w2 + b2, 'k--', label='SVC')
    plt.plot(x_lim, x_lim * w3 + b3, 'm', label='LinearSGD')
    plt.legend()
    plt.show()


def exercise9():
    dataset = datasets.fetch_mldata('MNIST original')
    X = dataset['data']
    y = dataset['target']

    dv = 60000
    X_train, X_test = X[:dv], X[dv:]
    y_train, y_test = y[:dv], y[dv:]
    rnd_idx = np.random.permutation(dv)
    X_train, y_train = X_train[rnd_idx], y_train[rnd_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # svc = LinearSVC(multi_class='ovr')
    svc = SVC(decision_function_shape='ovr')
    param_distr = {'gamma': reciprocal(0.001, 0.1), 'C': uniform(1, 10)}
    search_cv = RandomizedSearchCV(svc, param_distributions=param_distr, n_iter=10)
    search_cv.fit(X_train_scaled[:1000], y_train[:1000])

    y_test_pred = search_cv.predict(X_test_scaled)
    print(accuracy_score(y_test, y_test_pred))


def exercise10():
    dataset = datasets.fetch_california_housing()
    X = dataset['data']
    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR()
    params = {'gamma': reciprocal(0.001, 0.1), 'C': uniform(1, 10)}
    search_cv = RandomizedSearchCV(svr, param_distributions=params, n_iter=10)
    search_cv.fit(X_train_scaled, y_train)

    y_test_pred = search_cv.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_test_pred)
    print('rmse', np.sqrt(mse))
    print('best SVR', search_cv.best_estimator_)


exercise10()
