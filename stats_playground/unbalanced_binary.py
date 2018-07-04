from __future__ import division
import numpy as np

y = np.array([0] * 90 + [1] * 10)
# we predict 89 out of 90 `0` classes and only 1 out of 10 `1` classes
y_predict = np.array([0] * 89 + [1] + [0] * 8 + [1] * 2)

print('original y: {}'.format(y))
print('predicted y: {}'.format(y_predict))

correct_predictions = y == y_predict
accuracy = sum(correct_predictions) / correct_predictions.size
print('accuracy is: {}'.format(accuracy))

original_0 = y == 0
original_1 = y == 1

true_positive_of_0 = sum(y_predict[original_0] == 0)
false_negative_of_0 = sum(y_predict[original_0] == 1)
true_negative_of_0 = sum(y_predict[original_1] == 1)
false_positive_of_0 = sum(y_predict[original_1] == 0)

print('True positive of 0: {}'.format(true_positive_of_0))
print('False negative of 0: {}'.format(false_negative_of_0))
print('True negative of 0: {}'.format(true_negative_of_0))
print('False positive of 0: {}'.format(false_positive_of_0))

precision_of_0 = true_positive_of_0 / (true_positive_of_0 + false_positive_of_0)
print('Precision (fraction of relevant instances among retrieved) of 0: {}'.format(precision_of_0))
recall_of_0 = true_positive_of_0 / (true_positive_of_0 + false_negative_of_0)
print('Recall (fraction of relevant instances that have been retrieved among relevant) of 0: {}'.format(recall_of_0))

true_positive_of_1 = sum(y_predict[original_1] == 1)
false_negative_of_1 = sum(y_predict[original_1] == 0)
true_negative_of_1 = sum(y_predict[original_0] == 0)
false_positive_of_1 = sum(y_predict[original_0] == 1)

print('True positive of 1: {}'.format(true_positive_of_1))
print('False negative of 1: {}'.format(false_negative_of_1))
print('True negative of 1: {}'.format(true_negative_of_1))
print('False positive of 1: {}'.format(false_positive_of_1))

precision_of_1 = true_positive_of_1 / (true_positive_of_1 + false_positive_of_1)
print('Precision (fraction of relevant instances among retrieved) of 1: {}'.format(precision_of_1))
recall_of_1 = true_positive_of_1 / (true_positive_of_1 + false_negative_of_1)
print('Recall (fraction of relevant instances that have been retrieved among relevant) of 1: {}'.format(recall_of_1))
