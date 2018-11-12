import numpy as np
from scipy.stats import mode
from sklearn import datasets
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier


def find_best_estimator(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    estimator = DecisionTreeClassifier()
    # params_grid1 = {'max_leaf_nodes': [100, 300, 500, 700, 1000], 'max_depth': [5, 8, 10, 12]}
    # params_grid2 = {'max_leaf_nodes': np.arange(50, 160, 10), 'max_depth': [5, 6, 7, 8]}
    # params_grid3 = {'max_leaf_nodes': np.arange(20, 65, 5), 'max_depth': [5, 6, 7, 8]}
    # params_grid = {'max_leaf_nodes': np.arange(15, 30, 1), 'max_depth': [5, 6, 7]}
    params_grid = {'max_leaf_nodes': np.arange(15, 30, 1), 'max_depth': [5, 6, 7]}
    search_cv = GridSearchCV(estimator, params_grid, cv=5)
    search_cv.fit(X_train, y_train)
    print(search_cv.best_estimator_)
    y_test_pred = search_cv.best_estimator_.predict(X_test)
    print(accuracy_score(y_test, y_test_pred))


X, y = datasets.make_moons(n_samples=10000, noise=0.4, shuffle=True)
search_estimator = False
if search_estimator:
    find_best_estimator(X, y)
else:
    estimator = DecisionTreeClassifier(max_depth=6, max_leaf_nodes=23)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    n_subsets = 1000
    n_samples = 100
    subsets = []
    # create subsets
    splitter = ShuffleSplit(n_splits=n_subsets, train_size=n_samples)
    for subset_train_index, _ in splitter.split(X_train):
        X_subset_train = X_train[subset_train_index]
        y_subset_train = y_train[subset_train_index]
        subsets.append((X_subset_train, y_subset_train))
    # create forest of the best estimators
    forest = [clone(estimator) for _ in range(n_subsets)]
    accuracy_scores = []
    for tree, (X_subset_train, y_subset_train) in zip(forest, subsets):
        tree.fit(X_subset_train, y_subset_train)
        accuracy_scores.append(
            accuracy_score(y_test, tree.predict(X_test))
        )
    print('single tree: ', np.mean(accuracy_scores))

    y_test_pred = np.empty((n_subsets, X_test.shape[0]))
    for tree_idx, tree in enumerate(forest):
        y_test_pred[tree_idx] = tree.predict(X_test)
    y_test_pred_forest, n_votes = mode(y_test_pred, axis=0)
    print('forest:', accuracy_score(y_test, np.squeeze(y_test_pred_forest)))
