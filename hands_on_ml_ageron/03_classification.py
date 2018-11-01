import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import shift
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def plot_image(image_vector):
    plt.imshow(
        image_vector.reshape(28, 28),
        cmap=matplotlib.cm.binary,
        interpolation='nearest'
    )
    plt.axis('off')
    plt.show()


class ShiftImage4Times(BaseEstimator, TransformerMixin):
    def shift_image(self, image, dy_dx):
        image = image.reshape((28, 28))
        shifted_image = shift(image, dy_dx, cval=0, mode='constant')
        return shifted_image.reshape([-1])

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        X_transformed = []
        y_transformed = []

        for i in range(len(X)):
            x = X[i]
            X_transformed.append(x)
            y_transformed.append(self.y[i])
            for shift in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                X_transformed.append(self.shift_image(x, shift))
                y_transformed.append(self.y[i])

        return np.array(X_transformed), np.array(y_transformed)


def load_data():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    # we can't put shifter into 'prepare_pipeline' as it transform 'y' also.
    # shifter = ShiftImage4Times()
    # X_train, y_train = shifter.fit_transform(X_train, y_train)
    # X_test, y_test = shifter.fit_transform(X_test, y_test)
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    return X_train, y_train, X_test, y_test


def clf_confusion_matrix(y_classes, y, y_pred):
    mtrx = confusion_matrix(y, y_pred)
    mtrx_norm = mtrx / mtrx.sum(axis=1, keepdims=True)

    ax = sns.heatmap(mtrx_norm, annot=True, fmt='.2f', xticklabels=y_classes, yticklabels=y_classes)
    ax.xaxis.set_ticks_position('top')
    ax.figure.savefig('03_best_clf_conf_mtrx')


def clf_metrics(y, y_pred):
    def clf_metric(metric_func):
        metric_macro = metric_func(y, y_pred, average='macro')
        metric_weight = metric_func(y, y_pred, average='weighted')
        metric_micro = metric_func(y, y_pred, average='micro')
        print(
            '{} macro {}, weight {}, micro {}'.format(
                metric_func.__name__, metric_macro, metric_weight, metric_micro
            )
        )

    clf_metric(f1_score)
    clf_metric(precision_score)
    clf_metric(recall_score)
    print('accuracy {}'.format(accuracy_score(y, y_pred)))


X_train, y_train, X_test, y_test = load_data()

prepare_pipeline = Pipeline([
    ('scale', StandardScaler())
])
X_train = prepare_pipeline.fit_transform(X_train.astype(np.float64), y_train)
X_test = prepare_pipeline.transform(X_test.astype(np.float64))

param_distr = {
    'n_neighbors': randint(low=1, high=200),
    'weights': ['uniform', 'distance'],
}
# WARNING. This classifier is very heavy. I couldn't wait even one successful n_iter.
search_cv = RandomizedSearchCV(
    KNeighborsClassifier(),
    param_distr,
    n_iter=10, cv=3,
    scoring='f1_macro',
)

search_cv.fit(X_train, y_train)
best_clf = search_cv.best_estimator_
joblib.dump(best_clf, '03_best_clf.pkl')
y_classes = best_clf.classes_
y_pred = best_clf.predict(X_test)

clf_confusion_matrix(y_classes, y_test, y_pred)
clf_metrics(y_test, y_pred)
