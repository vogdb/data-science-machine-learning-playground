import os

import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def session_demo():
    x = tf.Variable(3, name='x')
    y = tf.Variable(3, name='y')
    f = x * x * y + y + 2

    init = tf.global_variables_initializer()
    use_interactive_sess = True
    if use_interactive_sess:
        sess = tf.InteractiveSession()
        init.run()
        result = f.eval()
        print(result)
        sess.close()
    else:
        with tf.Session() as sess:
            init.run()
            result = f.eval()
            print(result)


def graph_demo():
    use_default_graph = False
    if use_default_graph:
        x = tf.Variable(3, name='x')
        y = tf.Variable(3, name='y')
        f = x * x * y + y + 2
        print(x.graph is tf.get_default_graph())
    else:
        graph = tf.Graph()
        with graph.as_default():
            x = tf.Variable(3, name='x')
            y = tf.Variable(3, name='y')
            f = x * x * y + y + 2
        print(x.graph is tf.get_default_graph())
        print(x.graph is graph)


def linear_reg():
    housing_ds = fetch_california_housing()
    m, n = housing_ds.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing_ds.data]

    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
    y = tf.constant(housing_ds.target.reshape(-1, 1), dtype=tf.float32, name='y')
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()
        print(theta_value)


def gradient_descent():
    housing_ds = fetch_california_housing()
    m, n = housing_ds.data.shape
    scaler = StandardScaler()
    housing_data = scaler.fit_transform(housing_ds.data)
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing_data]

    n_epochs = 1000
    lr = 0.01
    batch_size = 100
    n_batches = int(np.ceil(m // batch_size))

    # X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
    # y = tf.constant(housing_ds.target.reshape(-1, 1), dtype=tf.float32, name='y')
    X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    # gradients = tf.gradients(mse, [theta])[0]
    # training_op = tf.assign(theta, theta - lr * gradients)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, '09_tensor_flow.ckpt')
        for epoch in range(n_epochs):
            for batch_i in range(n_batches):
                batch_idx_list = np.random.randint(0, m, size=batch_size)
                X_batch = housing_data_plus_bias[batch_idx_list]
                y_batch = housing_ds.target[batch_idx_list].reshape(-1, 1)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 100 == 0:
                saver.save(sess, '09_tensor_flow.ckpt')
                # print('Epoch {}, mse: {}'.format(epoch, mse.eval(feed_dict={X: housing_data_plus_bias, y: housing_ds.target.reshape(-1, 1)})))
        best_theta = theta.eval()
        print(best_theta)


def get_logdir():
    now = datetime.utcnow().strftime('%Y_%m_%d_%H:%M:%S')
    root_logdir = '09_tf_logs'
    logdir = '{}/run-{}/'.format(root_logdir, now)
    return logdir


def tensor_board():
    # data params
    housing_ds = fetch_california_housing()
    m, n = housing_ds.data.shape
    scaler = StandardScaler()
    housing_data = scaler.fit_transform(housing_ds.data)
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing_data]

    # train params
    n_epochs = 10
    lr = 0.01
    batch_size = 100
    n_batches = int(np.ceil(m // batch_size))

    # graph
    X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    with tf.name_scope('loss') as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(get_logdir(), tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_i in range(n_batches):
                batch_idx_list = np.random.randint(0, m, size=batch_size)
                X_batch = housing_data_plus_bias[batch_idx_list]
                y_batch = housing_ds.target[batch_idx_list].reshape(-1, 1)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                # write logs
                if batch_i % 10 == 0:
                    summary_val = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_i
                    file_writer.add_summary(summary_val, step)
        best_theta = theta.eval()
        print(best_theta)

    file_writer.close()


def modularity():
    def relu(X):
        # with tf.name_scope('relu'):
        with tf.variable_scope('relu', reuse=True):
            threshold = tf.get_variable('threshold')
            w_shape = (int(X.get_shape()[1]), 1)
            w = tf.Variable(tf.random_normal(w_shape), name='weights')
            b = tf.Variable(0.0, name='bias')
            z = tf.add(tf.matmul(X, w), b, name='z')
            return tf.maximum(z, threshold, name='relu')

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
    with tf.variable_scope('relu'):
        threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name='output')

    file_writer = tf.summary.FileWriter(get_logdir(), tf.get_default_graph())
    file_writer.close()


def exercise12():
    def generate_dataset():
        X, y = datasets.make_moons(n_samples=2000, shuffle=True, noise=0.2, random_state=42)
        y = y.reshape(-1, 1)
        X_plus_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return train_test_split(X_plus_bias, y, test_size=.2, shuffle=True)

    def random_batch(X, y, size):
        batch_idx = np.random.randint(0, X.shape[0], size)
        return X[batch_idx], y[batch_idx]

    def log_reg_graph(X, y, lr=0.01):
        with tf.variable_scope('log_reg', reuse=True):
            n = int(X.get_shape()[1])

            with tf.name_scope('model'):
                theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name='theta')
                q = tf.matmul(X, theta, name='q')
                y_proba = tf.sigmoid(q)  # 1 / (1 + tf.exp(-q))

            with tf.name_scope('loss'):
                epsilon = 1e-7  # to prevent from enormous logs
                log_loss = -tf.reduce_mean(
                    y * tf.log(y_proba + epsilon) + (1 - y) * tf.log(1 - y_proba + epsilon), name='cost'
                )
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
                training_op = optimizer.minimize(log_loss)
                loss_summary = tf.summary.scalar('loss', log_loss)

        return y_proba, training_op, loss_summary

    X_train, X_test, y_train, y_test = generate_dataset()
    m, n = X_train.shape
    batch_size = 100
    n_epochs = 100
    n_batches = int(np.ceil(m // batch_size))

    final_model_fp = '/tmp/09_exercise12_final_model'
    checkpoint_model_fp = '/tmp/09_exercise12_model.ckpt'
    checkpoint_epoch_fp = '/tmp/09_exercise12_model.epoch'

    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    y_proba, training_op, loss_summary = log_reg_graph(X, y)
    with tf.name_scope('init'):
        init = tf.global_variables_initializer()
    with tf.name_scope('save'):
        saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(get_logdir(), tf.get_default_graph())

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_fp):
            with open(checkpoint_epoch_fp, 'rb') as epoch_file:
                restore_epoch = int(epoch_file.read())
            saver.restore(sess, checkpoint_model_fp)
        else:
            sess.run(init)
            restore_epoch = 0
        for epoch in range(restore_epoch, n_epochs):
            for batch_i in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_summary_val = loss_summary.eval(feed_dict={X: X_test, y: y_test})
            file_writer.add_summary(loss_summary_val, epoch)
            if epoch % 10 == 0:
                saver.save(sess, checkpoint_model_fp)
                with open(checkpoint_epoch_fp, 'wb') as epoch_file:
                    epoch_file.write(b'%d' % epoch)

        saver.save(sess, final_model_fp)
        os.remove(checkpoint_epoch_fp)
        y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
        print('y_test_proba: {}'.format(y_proba_val))

    file_writer.close()


# session_demo()
# graph_demo()
# linear_reg()
# gradient_descent()
# tensor_board()
# modularity()
exercise12()
