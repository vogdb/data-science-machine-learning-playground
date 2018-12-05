import os
import re
import sys
import tarfile
from collections import defaultdict
from random import sample

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imresize
from six.moves import urllib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_sample_images
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.slim.nets import inception
from tensorflow.examples.tutorials.mnist import input_data

TF_MODELS_URL = 'http://download.tensorflow.org/models'
INCEPTION_V3_URL = TF_MODELS_URL + '/inception_v3_2016_08_28.tar.gz'
INCEPTION_PATH = os.path.join('datasets', 'inception')
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, 'inception_v3.ckpt')


def conv_demo():
    dataset = np.array(load_sample_images().images, dtype=np.float32)
    batch_size, height, width, channels = dataset.shape

    filters = np.zeros((7, 7, channels, 2), dtype=np.float32)
    filters[:, 3, :, 0] = 1  # vertical line
    filters[3, :, :, 1] = 1  # horizontal line

    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding='SAME')

    with tf.Session() as sess:
        output = sess.run(convolution, feed_dict={X: dataset})

    plt.imshow(output[0, :, :, 0])  # 1st image, 1st feature
    # plt.imshow(output[0, :, :, 1]) # 1st image, 2nd feature
    plt.axis('off')
    plt.show()


def pool_demo():
    dataset = np.array(load_sample_images().images, dtype=np.float64)
    batch_size, height, width, channels = dataset.shape

    X = tf.placeholder(tf.float64, shape=(None, height, width, channels))
    max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.Session() as sess:
        output = sess.run(max_pool, feed_dict={X: dataset})

    plt.figure(num=1)
    plt.imshow(dataset[0].astype(np.uint8))
    plt.axis('off')
    plt.figure(num=2)
    plt.imshow(output[0].astype(np.uint8))
    plt.axis('off')

    plt.show()


def exercise7():
    class MnistDNNClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, height, width):
            self.validation_fraction = .2
            self.learning_rate = .01
            self.n_epochs = 100
            self.batch_size = 20
            self._session = None
            self.height = height
            self.width = width

        def fit(self, X_train, y_train, X_valid, y_valid):
            self.close_session()

            max_n_epoch_no_progress = 20
            i_epoch_no_progress = 0
            best_loss_val = np.infty
            best_params = None

            n_outputs = len(np.unique(y_train))

            self._graph = tf.Graph()
            with self._graph.as_default():
                self._build_graph(n_outputs)

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
            return np.argmax(self.predict_proba(X), axis=1)

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

        def _build_graph(self, n_outputs):
            channels = 1
            n_inputs = self.height * self.width

            with tf.name_scope('inputs'):
                X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
                X_reshaped = tf.reshape(X, shape=[-1, self.height, self.width, channels])
                y = tf.placeholder(tf.int32, shape=(None), name='y')
                self._training = tf.placeholder_with_default(False, shape=(), name='training')

            with tf.name_scope('nn'):
                conv1 = tf.layers.conv2d(
                    X_reshaped, filters=32, kernel_size=3, strides=1,
                    padding='SAME', activation=tf.nn.relu, name='conv1'
                )
                conv2 = tf.layers.conv2d(
                    conv1, filters=64, kernel_size=3, strides=1,
                    padding='SAME', activation=tf.nn.relu, name='conv2'
                )
                with tf.name_scope('pool3'):
                    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                    pool3_flat = tf.reshape(pool3, shape=[-1, 64 * 14 * 14])
                    pool3_flat_drop = tf.layers.dropout(pool3_flat, .5, training=self._training)
                with tf.name_scope('fc1'):
                    fc1 = tf.layers.dense(pool3_flat_drop, 128, activation=tf.nn.relu, name='fc1')
                    fc1_drop = tf.layers.dropout(fc1, .5, training=self._training)
                with tf.name_scope('output'):
                    logits = tf.layers.dense(fc1_drop, n_outputs, name='logits')
                    Y_proba = tf.nn.softmax(logits, name='Y_proba')
            with tf.name_scope('train'):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                loss = tf.reduce_mean(xentropy, name='loss')
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                training_op = optimizer.minimize(loss, name='training_op')

            with tf.name_scope('eval'):
                correct = tf.nn.in_top_k(logits, y, 1)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
            with tf.name_scope('init_and_save'):
                self._init = tf.global_variables_initializer()
                self._saver = tf.train.Saver()

            self._X, self._y = X, y
            self._Y_proba, self._loss = Y_proba, loss
            self._training_op, self._accuracy = training_op, accuracy

    mnist = input_data.read_data_sets('/tmp/data/')
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_valid, y_valid = mnist.validation.images, mnist.validation.labels
    clf = MnistDNNClassifier(28, 28)
    clf.fit(X_train, y_train, X_valid, y_valid)


def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write('\rDownloading: {}%'.format(percent))
    sys.stdout.flush()


def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, 'inception_v3.tgz')
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)


def exercise8():
    CLASS_NAME_REGEX = re.compile(r'^n\d+\s+(.*)\s*$', re.M | re.U)

    def prepare_image(fname, show=False):
        channels = 3
        test_image = mpimg.imread(os.path.join(fname))[:, :, :channels]
        test_image = test_image.astype(np.float64)
        test_image_shape = test_image.shape
        if show:
            plt.imshow(test_image)
            plt.axis('off')
            plt.show()
        test_image = test_image.reshape(-1, 3)
        scaler = StandardScaler()
        test_image = scaler.fit_transform(test_image)
        return test_image.reshape(test_image_shape)

    def load_class_names():
        with open(os.path.join('datasets', 'inception', 'imagenet_class_names.txt'), 'rb') as f:
            content = f.read().decode('utf-8')
            return CLASS_NAME_REGEX.findall(content)

    test_img = prepare_image('cat.jpg', False)
    # fetch_pretrained_inception_v3()
    class_names = ['background'] + load_class_names()

    X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)
    predictions = end_points['Predictions']
    saver = tf.train.Saver()

    X_test = test_img.reshape((-1,) + test_img.shape)
    with tf.Session() as sess:
        saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
        predictions_val = sess.run(predictions, feed_dict={X: X_test})
        top5 = np.argpartition(predictions_val[0], -5)[-5:]
        top5 = reversed(top5[np.argsort(predictions_val[0][top5])])
        for i in top5:
            print('{0}: {1:.2f}%'.format(class_names[i], 100 * predictions_val[0][i]))


def exercise9():
    FLOWERS_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
    FLOWERS_PATH = os.path.join('datasets', 'flowers')
    HEIGHT = 299
    WIDTH = 299
    CHANNELS = 3

    def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
        if os.path.exists(FLOWERS_PATH):
            return
        os.makedirs(path, exist_ok=True)
        tgz_path = os.path.join(path, 'flower_photos.tgz')
        urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
        flowers_tgz = tarfile.open(tgz_path)
        flowers_tgz.extractall(path=path)
        flowers_tgz.close()
        os.remove(tgz_path)

    def read_flowers_classes(flowers_path):
        return sorted([
            dirname for dirname in os.listdir(flowers_path)
            if os.path.isdir(os.path.join(flowers_path, dirname))
        ])

    def image_paths_by_class(flowers_path, flower_classes):
        image_paths = defaultdict(list)

        for flower_class in flower_classes:
            image_dir = os.path.join(flowers_path, flower_class)
            for filepath in os.listdir(image_dir):
                if filepath.endswith('.jpg'):
                    image_paths[flower_class].append(os.path.join(image_dir, filepath))

        return image_paths

    def get_flower_paths_and_classes(image_paths, flower_class_ids):
        flower_paths_and_classes = []
        for flower_class, paths in image_paths.items():
            for path in paths:
                flower_paths_and_classes.append((path, flower_class_ids[flower_class]))
        np.random.shuffle(flower_paths_and_classes)
        return flower_paths_and_classes

    def prepare_image(image, target_width=299, target_height=299):
        # resize the image to the target dimensions.
        image = imresize(image, (target_width, target_height))
        # ensure that the colors are represented as 32-bit floats ranging from 0.0 to 1.0
        return image.astype(np.float32) / 255

    def prepare_batch(flower_paths_and_classes, batch_size):
        batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
        images = [prepare_image(mpimg.imread(path)[:, :, :CHANNELS]) for path, labels in batch_paths_and_classes]
        X_batch = 2 * np.stack(images) - 1  # Inception expects colors ranging from -1 to 1
        y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
        return X_batch, y_batch

    flowers_root_path = os.path.join(FLOWERS_PATH, 'flower_photos')
    fetch_flowers()
    flower_classes = read_flowers_classes(flowers_root_path)
    flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
    image_paths = image_paths_by_class(flowers_root_path, flower_classes)
    flower_paths_and_classes = get_flower_paths_and_classes(image_paths, flower_class_ids)

    test_ratio = 0.2
    train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))
    flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
    flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

    n_outputs = len(flower_classes)
    X = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNELS])
    y = tf.placeholder(tf.int32, shape=[None])
    training = tf.placeholder_with_default(False, shape=[])

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)
    inception_saver = tf.train.Saver()
    # squeeze the shape (?, 1, 1, 2048) => (?, 2048)
    prelogits = tf.squeeze(end_points['PreLogits'], axis=[1, 2])

    with tf.name_scope('new_output_layer'):
        flower_logits = tf.layers.dense(prelogits, n_outputs, name='flower_logits')
        Y_proba = tf.nn.softmax(flower_logits, name='Y_proba')
    with tf.name_scope('train'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flower_logits')
        training_op = optimizer.minimize(loss, var_list=flower_vars)
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(flower_logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope('init_and_save'):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    X_test, y_test = prepare_batch(flower_paths_and_classes_test, len(flower_paths_and_classes_test))
    batch_size = 40
    n_batches = len(flower_paths_and_classes_train) // batch_size
    with tf.Session() as sess:
        init.run()
        inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

        for epoch in range(100):
            for batch_i in range(n_batches):
                X_batch, y_batch = prepare_batch(flower_paths_and_classes, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

            accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print('accuracy: {}'.format(accuracy_val))


exercise9()
