import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def exercise8_9():
    mnist = datasets.fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, shuffle=True)
    X_train, y_train = X_train[:5000], y_train[:5000]

    svc = LinearSVC(random_state=42)
    rnd_for = RandomForestClassifier(n_estimators=10, random_state=42)
    extr_forest = ExtraTreesClassifier(n_estimators=10, random_state=42)
    mlp_clf = MLPClassifier(random_state=42)
    voting_clf = VotingClassifier(
        estimators=[('svm', svc), ('forest', rnd_for), ('extra forest', extr_forest), ('mlp', mlp_clf)],
        voting='hard'
    )
    voting_clf.fit(X_train, y_train)
    estimator_score_list = [estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
    print('Hard score {}, sep score {}'.format(voting_clf.score(X_val, y_val), estimator_score_list))

    voting_clf.set_params(svm=None, voting='soft')
    del voting_clf.estimators_[0]
    estimator_score_list = [estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
    print('Soft score {}, sep score {}'.format(voting_clf.score(X_val, y_val), estimator_score_list))

    X_val_pred = np.array([estimator.predict(X_val) for estimator in voting_clf.estimators_]).T
    blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
    blender.fit(X_val_pred, y_val)

    X_test_pred = np.array([estimator.predict(X_test) for estimator in voting_clf.estimators_]).T
    print('Blender score {}'.format(blender.score(X_test_pred, y_test)))


exercise8_9()
