import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def ensemble_demo():
    X, y = datasets.make_moons(n_samples=10000, noise=0.2, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    log_reg = LogisticRegression(solver='lbfgs')
    svc = SVC(gamma='scale', probability=True)
    rnd_for = RandomForestClassifier(n_estimators=10)

    voting_clf = VotingClassifier(
        estimators=[('log reg', log_reg), ('svm', svc), ('forest', rnd_for)],
        voting='soft'
    )

    for clf in [log_reg, svc, rnd_for, voting_clf]:
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        print(type(clf).__name__, accuracy_score(y_test, y_test_pred))


def bagging_demo():
    X, y = datasets.make_moons(n_samples=10000, noise=0.2, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1
    )
    bag_clf.fit(X_train, y_train)
    y_test_pred = bag_clf.predict(X_test)
    print(accuracy_score(y_test, y_test_pred))


def bagging_oob_demo():
    X, y = datasets.make_moons(n_samples=10000, noise=0.2, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500, bootstrap=True, n_jobs=-1, oob_score=True
    )
    bag_clf.fit(X_train, y_train)
    y_test_pred = bag_clf.predict(X_test)
    print('oob score', bag_clf.oob_score_)
    print('accuracy_score', accuracy_score(y_test, y_test_pred))
    print('oob decision function\n', bag_clf.oob_decision_function_)


def feature_importance():
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)
    rnd_for = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rnd_for.fit(X_train, y_train)
    print('RandomForest', accuracy_score(y_test, rnd_for.predict(X_test)))
    for name, importance in zip(dataset['feature_names'], rnd_for.feature_importances_):
        print(name, importance)

    extr_for = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
    extr_for.fit(X_train, y_train)
    print('ExtremeRandomForest', accuracy_score(y_test, extr_for.predict(X_test)))
    for name, importance in zip(dataset['feature_names'], extr_for.feature_importances_):
        print(name, importance)


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs', alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18, rotation=0)


def ada_boost_manual():
    X, y = datasets.make_moons(n_samples=500, noise=0.30, random_state=42)
    m = len(X)

    plt.figure(figsize=(11, 4))
    for subplot, learning_rate in ((121, 1), (122, 0.5)):
        sample_weights = np.ones(m)
        plt.subplot(subplot)
        for i in range(4):
            svm_clf = SVC(gamma='auto', kernel='rbf', C=0.05, random_state=42)
            svm_clf.fit(X, y, sample_weight=sample_weights)
            y_pred = svm_clf.predict(X)
            sample_weights[y_pred != y] *= (1 + learning_rate)
            plot_decision_boundary(svm_clf, X, y, alpha=0.2)
            plt.title('learning_rate = {}'.format(learning_rate), fontsize=16)


def ada_boost():
    X, y = datasets.make_moons(n_samples=500, noise=0.30, random_state=42)
    svm_clf = SVC(gamma='auto', kernel='rbf', C=0.05, probability=True, random_state=42)
    ada_clf = AdaBoostClassifier(base_estimator=svm_clf, n_estimators=10, learning_rate=1, random_state=42)
    # ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate=.5, random_state=42)
    ada_clf.fit(X, y)
    plot_decision_boundary(ada_clf, X, y, alpha=0.2)


def gradient_boost_manual():
    def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
        x1 = np.linspace(axes[0], axes[1], 500)
        y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
        plt.plot(X[:, 0], y, data_style, label=data_label)
        plt.plot(x1, y_pred, style, linewidth=2, label=label)
        if label or data_label:
            plt.legend(loc="upper center", fontsize=16)
        plt.axis(axes)

    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    plt.plot(X, y, 'b.')

    tree_reg1 = DecisionTreeRegressor(max_depth=2)
    tree_reg1.fit(X, y)

    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2)
    tree_reg2.fit(X, y2)

    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2)
    tree_reg3.fit(X, y3)

    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 1.0], style='c:', label='1')
    plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 1.0], style='g:', label='2')
    plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 1.0], style='r--', label='3')
    plt.legend()


def gradient_boost_early_stop():
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    staged_predict = False
    if staged_predict:
        grbr = GradientBoostingRegressor(max_depth=2, n_estimators=120)
        grbr.fit(X_train, y_train)

        errors = [mean_squared_error(y_test, y_test_pred) for y_test_pred in grbr.staged_predict(X_test)]
        best_n_estimators = np.argmin(errors)
        best_grbr = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
        best_grbr.fit(X_train, y_train)
        print('staged_predict n_estimators: {}, mse {}'.format(best_n_estimators, np.amin(errors)))
        print('staged_predict mse ', np.amin(errors))
    else:
        grbr = GradientBoostingRegressor(max_depth=2, warm_start=True)
        error_increase = 0
        n_estimators = 1
        min_test_error = float('inf')
        for n_estimators in range(1, 120):
            grbr.n_estimators = n_estimators
            grbr.fit(X_train, y_train)
            y_test_pred = grbr.predict(X_test)
            test_error = mean_squared_error(y_test, y_test_pred)
            if test_error < min_test_error:
                min_test_error = test_error
                error_increase = 0
            else:
                error_increase += 1
                if error_increase > 5:
                    break
        print('warm_start n_estimators: {}, mse {}'.format(n_estimators, min_test_error))


gradient_boost_early_stop()
plt.show()
