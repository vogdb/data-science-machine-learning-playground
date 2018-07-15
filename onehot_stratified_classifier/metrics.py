import keras as keras
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        self._data.append(self.calculate(self.model, X_val, y_val))
        return

    def calculate(self, model, X_val, y_val):
        y_predict = np.asarray(model.predict(X_val))
        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        result = {
            'val_acc': accuracy_score(y_val, y_predict),
            'val_recall': recall_score(y_val, y_predict),
            'val_precision': precision_score(y_val, y_predict),
            'val_f1': f1_score(y_val, y_predict),
            'val_roc_auc': roc_auc_score(y_val, y_predict),
        }
        return result

    def get_data(self):
        return self._data
