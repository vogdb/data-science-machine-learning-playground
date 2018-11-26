import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def dnn_tflearn_demo():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    w, h = X_train.shape[1], X_train.shape[2]
    X_train = X_train.astype(np.float32).reshape(-1, w * h) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, w * h) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    feature_cols = [tf.feature_column.numeric_column('X', shape=[w * h])]
    dnn_clf = tf.estimator.DNNClassifier(
        hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols
    )
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True
    )
    dnn_clf.train(input_fn=input_fn)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': X_test}, y=y_test, shuffle=False
    )
    eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
    print(eval_results)


def dnn_plaintf_demo():
    def neuron_layer(X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name='weights')
            b = tf.Variable(tf.zeros([n_neurons]), name='biases')
            Z = tf.matmul(X, W) + b
            if activation is None:
                return Z
            else:
                return activation(Z)

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    lr = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    with tf.name_scope('dnn'):
        hidden1 = neuron_layer(X, n_hidden1, 'hidden1', activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, 'hidden2', activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, 'outputs')
        # hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
        # hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
        # logits = tf.layers.dense(hidden2, n_outputs, name='outputs')
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        training_op = optimizer.minimize(loss)
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # execution
    mnist = input_data.read_data_sets('/tmp/data')
    n_epochs = 100
    batch_size = 50
    n_batches = int(np.ceil(mnist.train.num_examples // batch_size))

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for batch_i in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
            print('acc_test {}'.format(acc_test))
        saver.save(sess, './10_dnn_tf_plain_model.ckpt')

    with tf.Session() as sess:
        saver.restore(sess, './10_dnn_tf_plain_model.ckpt')
        X_new = mnist.test.images
        Z = logits.eval(feed_dict={X: X_new})
        y_pred = np.argmax(Z, axis=1)


def exercise9():
    def mlp_graph(X, y, n_hidden1, n_hidden2, n_outputs, lr):
        with tf.name_scope('mlp'):
            with tf.name_scope('layers'):
                hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
                hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
                logits = tf.layers.dense(hidden2, n_outputs, name='outputs')
            with tf.name_scope('loss'):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                loss = tf.reduce_mean(xentropy, name='loss')
                loss_summary = tf.summary.scalar('loss', loss)
            with tf.name_scope('train'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
                training_op = optimizer.minimize(loss)
        return training_op, logits, xentropy, loss_summary

    X = tf.placeholder(tf.float32, shape=(None, 28 * 28), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    training_op, logits, xentropy, loss_summary = mlp_graph(X, y, 300, 100, 10, 0.01)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter('10_tf_logs', tf.get_default_graph())

    mnist = input_data.read_data_sets('/tmp/data')
    n_epochs = 10
    batch_size = 50
    n_batches = int(np.ceil(mnist.train.num_examples // batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_i in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_summary_val = loss_summary.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
            file_writer.add_summary(loss_summary_val, epoch)

        saver.save(sess, './10_dnn_tf_ex9_model.ckpt')

        y_pred_logits = logits.eval(feed_dict={X: mnist.test.images})
        y_pred = np.argmax(y_pred_logits, axis=1)
        print('test accuracy'.format(np.mean(y_pred == mnist.test.labels)))

    file_writer.close()


exercise9()
