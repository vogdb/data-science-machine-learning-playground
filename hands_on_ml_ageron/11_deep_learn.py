import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split


def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return X_train, X_test, y_train, y_test


def cut_data(digit):
    X_train, X_test, y_train, y_test = load_data()
    X_train = X_train[y_train < digit]
    y_train = y_train[y_train < digit]
    X_test = X_test[y_test < digit]
    y_test = y_test[y_test < digit]
    return X_train, X_test, y_train, y_test


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, n_hidden_layers=5, n_neurons=100, n_epochs=100, activation=tf.nn.elu, lr=0.01, batch_size=20,
            validation_fraction=.1, optimizer_class=tf.train.AdamOptimizer, initializer=None,
            batch_norm_momentum=None, dropout_rate=None, random_state=None
    ):
        if initializer is None:
            initializer = tf.variance_scaling_initializer()

        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.validation_fraction = validation_fraction
        self.optimizer_class = optimizer_class
        self.learning_rate = lr
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self._session = None

    def fit(self, X, y):
        self.close_session()

        max_n_epoch_no_progress = 20
        i_epoch_no_progress = 0
        best_loss_val = np.infty
        best_params = None

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.validation_fraction)
        self._classes = np.unique(y)
        n_inputs = X.shape[1]
        n_outputs = len(self._classes)
        self._class_to_index = {label: index for index, label in enumerate(self._classes)}
        y = np.array([self._class_to_index[label] for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(self.n_epochs):
                rnd_idx = np.random.permutation(len(X_train))
                for batch_idx in np.array_split(rnd_idx, int(len(X_train) // self.batch_size)):
                    X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                loss_val, acc_val = sess.run(
                    [self._loss, self._accuracy], feed_dict={self._X: X_valid, self._y: y_valid}
                )
                if loss_val < best_loss_val:
                    best_params = self._get_model_params()
                    best_loss_val = loss_val
                    i_epoch_no_progress = 0
                else:
                    i_epoch_no_progress += 1
                    if i_epoch_no_progress > max_n_epoch_no_progress:
                        print('Early stopping at epoch {}'.format(epoch))
                        break
            if best_params:
                self._restore_model_params(best_params)
            return self

    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {
            gvar_name: self._graph.get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names
        }
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self._classes[class_index] for class_index in class_indices])

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError('This %s instance is not fitted yet' % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def save(self, path):
        self._saver.save(self._session, path)

    def close_session(self):
        if self._session:
            self._session.close()

    def _dnn(self, input):
        with tf.variable_scope('dnn'):
            for l in range(self.n_hidden_layers):
                if self.dropout_rate:
                    input = tf.layers.dropout(input, self.dropout_rate, training=self._training)
                input = tf.layers.dense(
                    input, self.n_neurons, name='hidden%d' % (l + 1),
                    kernel_initializer=self.initializer
                )
                if self.batch_norm_momentum:
                    input = tf.layers.batch_normalization(
                        input, momentum=self.batch_norm_momentum, training=self._training
                    )
                input = self.activation(input, name='hidden%d_out' % (l + 1))
            return input

    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.set_random_seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
        y = tf.placeholder(tf.int32, shape=(None), name='y')
        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        dnn_output = self._dnn(X)

        logits = tf.layers.dense(
            dnn_output, n_outputs, kernel_initializer=tf.variance_scaling_initializer(), name='logits'
        )
        Y_proba = tf.nn.softmax(logits, name='Y_proba')
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')

        optimizer = self.optimizer_class(self.learning_rate)
        training_op = optimizer.minimize(loss, name='training_op')

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        self._init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy


def exercise8():
    X_train, X_test, y_train, y_test = cut_data(5)

    dnn_clf = DNNClassifier(batch_norm_momentum=.9, dropout_rate=.5)
    dnn_clf.fit(X_train, y_train)


exercise8()
