import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def plot_image(image, shape=(28, 28)):
    plt.imshow(image.reshape(shape), cmap='Greys', interpolation='nearest')
    plt.axis('off')


def plot_reconstructed_images(X, outputs, X_test, saver, model_path):
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        outputs_val = outputs.eval(feed_dict={X: X_test})

    n_test_samples = X_test.shape[0]
    plt.figure(figsize=(8, 3 * n_test_samples))
    for sample_idx in range(n_test_samples):
        plt.subplot(n_test_samples, 2, sample_idx * 2 + 1)
        plot_image(X_test[sample_idx])
        plt.subplot(n_test_samples, 2, sample_idx * 2 + 2)
        plot_image(outputs_val[sample_idx])


def exercise8():
    def plot_images():
        is_plot_weights = True
        if is_plot_weights:
            with tf.Session() as sess:
                saver.restore(sess, model_filename)
                weights = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden1.name)[0] + '/kernel:0')
                # (768, 300)
                weights_val = weights.eval()
                for i in range(1, 10):
                    plt.subplot(3, 3, i)
                    plot_image(weights_val.T[i])
        else:
            X_test = mnist.test.images[:2]
            plot_reconstructed_images(X, outputs, X_test, saver, model_filename)
        plt.show()

    mnist = input_data.read_data_sets('/tmp/data/')
    X_train, y_train = mnist.train.images, mnist.train.labels

    n_train_samples, n_inputs = X_train.shape
    n_outputs = n_inputs
    n_hidden1 = 300
    n_hidden2 = 150
    n_hidden3 = n_hidden1
    batch_size = 150
    n_epochs = 20
    noise_level = 1.
    l2_reg = .001
    lr = .01
    model_filename = './15_ex8_model'

    X = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.int32, [None])
    X_noisy = X + noise_level * tf.random_normal(tf.shape(X))

    dense_layer = partial(
        tf.layers.dense,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
    )

    hidden1 = dense_layer(X_noisy, n_hidden1, name='hidden1')
    hidden2 = dense_layer(hidden1, n_hidden2, name='hidden2')

    # autoencoder
    hidden3 = dense_layer(hidden2, n_hidden3, name='hidden3')
    outputs = tf.layers.dense(hidden3, n_outputs, name='outputs')
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    autoenc_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    autoenc_training_op = autoenc_optimizer.minimize(reconstruction_loss)

    # classifier
    logits = tf.layers.dense(hidden2, 10, name='logits')
    with tf.variable_scope('logits', reuse=True):
        logits_weights = tf.get_variable('kernel')
        logits_biases = tf.get_variable('bias')
    clf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    clf_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    clf_training_op = clf_optimizer.minimize(clf_loss, var_list=[logits_weights, logits_biases])
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    best_loss_val = np.infty
    with tf.Session() as sess:
        init.run()
        X_batch = None
        for epoch in range(n_epochs):
            for batch_i in range(n_train_samples // batch_size):
                X_batch, _ = mnist.train.next_batch(batch_size)
                sess.run(autoenc_training_op, feed_dict={X: X_batch})
            loss_val = reconstruction_loss.eval(feed_dict={X: X_batch})
            print('epoch: {}, loss: {}'.format(epoch, loss_val))
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                saver.save(sess, model_filename)

    is_plot_images = False
    if is_plot_images:
        plot_images()
    else:
        with tf.Session() as sess:
            saver.restore(sess, model_filename)
            for epoch in range(n_epochs):
                for batch_i in range(n_train_samples // batch_size):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(clf_training_op, feed_dict={X: X_batch, y: y_batch})
                accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
                print('epoch: {}, accuracy: {}'.format(epoch, accuracy_val))


exercise8()
