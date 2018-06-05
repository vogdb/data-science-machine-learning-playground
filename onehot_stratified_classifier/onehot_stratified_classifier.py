import keras
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from create_model import create_model

train_params = dict(
    batch_size=128,
    epochs=200,
    fold_num=10,
    verbose=0
)

x = np.random.random((1000, 20))
y = np.array([0] * 500 + [1] * 500)


def stratified_fold_cycle(x, y):
    kfold = StratifiedKFold(n_splits=train_params['fold_num'], shuffle=True)
    kfold_split = kfold.split(x, y)
    y = keras.utils.to_categorical(y, num_classes=2)
    val_acc_list = []

    for train_indices, val_indices in kfold_split:
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_val = x[val_indices]
        y_val = y[val_indices]
        model = create_model()
        model.fit(
            x_train,
            y_train,
            epochs=train_params['epochs'],
            batch_size=train_params['batch_size'],
            verbose=train_params['verbose'],
        )
        score = model.evaluate(x_val, y_val)
        val_acc_list.append(score[1])
    return val_acc_list


def plain_cross_val_score(x, y):
    y = keras.utils.to_categorical(y, num_classes=2)
    estimator = KerasClassifier(
        build_fn=create_model,
        epochs=train_params['epochs'],
        batch_size=train_params['batch_size'],
        verbose=train_params['verbose'],
    )
    kfold = KFold(n_splits=train_params['fold_num'])
    return cross_val_score(estimator, x, y, cv=kfold)


def stratified_cross_val_score(x, y):
    kfold = StratifiedKFold(n_splits=train_params['fold_num'], shuffle=True)
    kfold_split = kfold.split(x, y)
    y = keras.utils.to_categorical(y, num_classes=2)
    x_stratified = []
    y_stratified = []

    for train_indices, val_indices in kfold_split:
        x_val = x[val_indices]
        y_val = y[val_indices]
        x_stratified.append(x_val)
        y_stratified.append(y_val)
    x_stratified = np.concatenate(tuple(x_stratified))
    y_stratified = np.concatenate(tuple(y_stratified))

    estimator = KerasClassifier(
        build_fn=create_model,
        epochs=train_params['epochs'],
        batch_size=train_params['batch_size'],
        verbose=train_params['verbose'],
    )
    kfold = KFold(n_splits=train_params['fold_num'])
    # results = cross_val_score(estimator, X, y, cv=kfold, fit_params={'callbacks': [metrics]})
    # Important! `metrics` requires `validation_data` which can't be figured out in cross_val_score
    return cross_val_score(estimator, x_stratified, y_stratified, cv=kfold)


def print_result(acc_list, header):
    print('############################################')
    print(header)
    print(['{0:0.2f}'.format(acc) for acc in acc_list])
    acc_list = np.array(acc_list)
    print('Mean:', acc_list.mean(), 'STD:', acc_list.std())
    print('--------------------------------------------')


stratified_cycle_result = stratified_fold_cycle(x, y)
plain_cross_result = plain_cross_val_score(x, y)
stratified_cross_result = stratified_cross_val_score(x, y)

print_result(stratified_cycle_result, 'Stratified KFold cycle validation accuracy')
print_result(plain_cross_result, 'Plain KFold cross_val_score validation accuracy')
print_result(stratified_cross_result, 'Stratified KFold cross_val_score validation accuracy')
