## About

This demo shows how to apply `StratifiedKFold` to one-hot labels with `KerasClassifier` and `scikit.cross_val_score`.

### Result
```
############################################
Stratified KFold cycle validation accuracy
['0.49', '0.51', '0.56', '0.48', '0.51', '0.53', '0.44', '0.54', '0.52', '0.48']
Mean: 0.506 STD: 0.03292415526630867
--------------------------------------------
############################################
Plain KFold cross_val_score validation accuracy
['0.00', '0.07', '0.01', '0.00', '0.00', '0.00', '0.00', '0.03', '0.00', '0.01']
Mean: 0.011999999918043614 STD: 0.021354156532672867
--------------------------------------------
############################################
Stratified KFold cross_val_score validation accuracy
['0.56', '0.49', '0.55', '0.48', '0.45', '0.46', '0.51', '0.57', '0.49', '0.50']
Mean: 0.5060000002384186 STD: 0.0392937664645392
--------------------------------------------
```