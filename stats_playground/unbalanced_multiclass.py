from __future__ import division
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd

np.set_printoptions(precision=2)

y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 2])
y_pred = np.array([0, 2, 2, 0, 0, 1, 1, 0, 0])
labels = np.array(list(set(y_true)))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
cnf_matrix_norm = np.round(cnf_matrix_norm, 2)


def print_cnf_matrix(cnf_matrix):
    labels_true = np.core.defchararray.add('true ', labels.astype(str)[:, np.newaxis])
    cnf_matrix = np.hstack((labels_true, cnf_matrix))
    labels_pred = np.core.defchararray.add('pred ', labels.astype(str))
    cnf_matrix = np.vstack((np.concatenate([[''], labels_pred]), cnf_matrix))
    df = pd.DataFrame(cnf_matrix)
    print(df.to_string(index=False))


print('true y: {}'.format(y_true))
print('pred y: {}'.format(y_pred))
print('Confusion matrix without normalization:')
print_cnf_matrix(cnf_matrix)
print('Confusion matrix normalized:')
print_cnf_matrix(cnf_matrix_norm)


def precision_of_class(class_label):
    class_true_idx = y_true == class_label
    correct_class_pred = sum(y_pred[class_true_idx] == class_label)
    general_class_pred = sum(y_pred == class_label)
    return correct_class_pred / general_class_pred


def precision_of_class_by_conf_matrix(class_label):
    class_label_idx = np.where(labels == class_label)
    correct_class_pred = cnf_matrix[class_label_idx, class_label_idx]
    general_class_pred = np.sum(cnf_matrix[:, class_label_idx])
    return correct_class_pred / general_class_pred


print('Accuracy is: {}'.format(metrics.accuracy_score(y_true, y_pred)))
precision_per_class = [precision_of_class(class_label) for class_label in labels]
print('Precision per class: {}'.format(precision_per_class))
prec_none = metrics.precision_score(y_true, y_pred, average=None)
print('The same by scipy (average=None): {}'.format(prec_none))
precision_per_class = [precision_of_class_by_conf_matrix(class_label) for class_label in labels]
print('The same by confusion matrix: {}'.format(prec_none))

print('Precision macro. Mean of precisions per class: {}'.format((1 / len(labels)) * sum(precision_per_class)))
prec_macro = metrics.precision_score(y_true, y_pred, average='macro')
print('The same by scipy (average="macro"): {0:.2f}'.format(prec_macro, 2))

prec_weighted = sum(
    [(sum(y_true == label) / len(y_true)) * precision_per_class[idx] for idx, label in enumerate(labels)]
)
print(
    'Precision weighted. Sum of precisions per class where each class\' precision is weighted by its presence in true data. Major classes have more weight and vice versa:\n{}'
        .format(prec_weighted)
)
prec_weighted = metrics.precision_score(y_true, y_pred, average='weighted')
print('The same by scipy (average="weighted"): {0:.2f}'.format(prec_weighted, 2))

prec_micro = metrics.precision_score(y_true, y_pred, average='micro')
print(
    'Precision micro. It sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient. It\'s the same as accuracy :\n{0:.2f}'
        .format(prec_micro)
)
