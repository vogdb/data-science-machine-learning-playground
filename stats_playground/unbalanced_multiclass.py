import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

np.set_printoptions(precision=2)
pd.set_eng_float_format(accuracy=3)

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.4)
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cnf_matrix_norm = np.round(cnf_matrix_norm, 2)


def print_cnf_matrix(cnf_matrix):
    labels = np.array(iris.target_names)
    labels_true = np.core.defchararray.add('true ', labels[:, np.newaxis])
    cnf_matrix = np.hstack((labels_true, cnf_matrix))
    labels_pred = np.core.defchararray.add('pred ', labels)
    cnf_matrix = np.vstack((np.concatenate([[''], labels_pred]), cnf_matrix))
    df = pd.DataFrame(cnf_matrix)
    print(df.to_string())


print('Confusion matrix without normalization:')
print_cnf_matrix(cnf_matrix)
print('Confusion matrix normalized:')
print_cnf_matrix(cnf_matrix_norm)
