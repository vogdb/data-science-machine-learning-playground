import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def plain_rnn():
    n_neurons = 5
    n_inputs = 3

    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])

    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
    b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

    init = tf.global_variables_initializer()

    X0_batch = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 0, 1],
    ])
    X1_batch = np.array([
        [9, 8, 7],
        [0, 0, 0],
        [6, 5, 4],
        [3, 2, 1]
    ])

    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

    print(Y0_val)
    print(Y1_val)


def static_rnn():
    n_neurons = 5
    n_inputs = 3

    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])

    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

    Y0, Y1 = output_seqs
    print(Y0, Y1)


def dynamic_rnn():
    n_inputs = 3
    n_neurons = 5

    X = tf.placeholder(tf.float32, [None, None, n_inputs])
    seq_length = tf.placeholder(tf.int32, [None])
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
    init = tf.global_variables_initializer()
    file_writer = tf.summary.FileWriter('14_tf_logs', tf.get_default_graph())
    file_writer.close()

    X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]],  # instance 1
        [[3, 4, 5], [0, 0, 0]],  # instance 2
        [[3, 4, 5], [3, 6, 1]],  # instance 2
    ])
    seq_length_batch = np.array([2, 1, 2])
    with tf.Session() as sess:
        init.run()
        outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

    print('Outputs:')
    print(outputs_val)
    print('States:')
    print(states_val)


def mnist_rnn():
    n_steps = 28  # each image row is a time step
    n_inputs = 28  # each pixel of row is an input
    n_neurons = 150
    n_outputs = 10

    lr = 0.001

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = tf.layers.dense(states, n_outputs, name='logits')
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()

    mnist = input_data.read_data_sets('/tmp/data/')
    X_test = mnist.test.images.reshape(-1, n_steps, n_inputs)
    y_test = mnist.test.labels

    n_epochs = 100
    batch_size = 200

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for batch_i in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_test_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
            print('Epoch: {}, Test accuracy: {}'.format(epoch, acc_test_val))


def predict_stock():
    def time_series(t):
        return t * np.sin(t) / 3 + 2 * np.sin(t * 5)

    def next_batch(batch_size, n_steps):
        t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
        Ts = t0 + np.arange(0., n_steps + 1) * resolution
        ys = time_series(Ts)
        return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

    t_min, t_max = 0, 30
    resolution = 0.1
    t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
    # t = np.arange(t_min, t_max + resolution, resolution)

    n_steps = 20
    n_outputs = 1
    n_neurons = 100
    n_inputs = 1
    use_projection_wrapper = False

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    if use_projection_wrapper:
        cell = rnn.OutputProjectionWrapper(
            rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs
        )
        outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
        outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 1500
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 100 == 0:
                mse_val = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print('Epoch: {}, mse: {}'.format(epoch, mse_val))
        saver.save(sess, './14_stock_predict_model')

    with tf.Session() as sess:
        saver.restore(sess, './14_stock_predict_model')

        is_predict_sequence = True
        if is_predict_sequence:
            seq_len = 300
            seq = np.zeros(n_steps, dtype=np.float32)
            for i in range(seq_len):
                X_batch = seq[-n_steps:].reshape(1, n_steps, 1)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
                seq = np.append(seq, y_pred[0, -1, 0])
            plt.plot(seq, 'b-')
            plt.xlabel('Time')
        else:
            t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
            X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
            y_pred = sess.run(outputs, feed_dict={X: X_new})
            print(X_new.shape, y_pred.shape)
            plt.plot(t_instance[:-1], time_series(t_instance[:-1]), 'bo', markersize=10, label='instance')
            plt.plot(t_instance[1:], time_series(t_instance[1:]), 'y*', markersize=10, label='target')
            plt.plot(t_instance[1:], y_pred[0, :, 0], 'r.', markersize=10, label='prediction')
            plt.legend()

        plt.show()


def exercise7():
    grammar_lvl1 = [
        [('B', 1)],
        [('T', 2), ('P', 3)],
        [('S', 2), ('X', 4)],
        [('T', 3), ('V', 5)],
        [('X', 3), ('S', 6)],
        [('P', 4), ('V', 6)],
        [('E', None)]
    ]

    grammar_lvl2 = [
        [('B', 1)],
        [('T', 2), ('P', 3)],
        [(grammar_lvl1, 4)],
        [(grammar_lvl1, 5)],
        [('T', 6)],
        [('P', 6)],
        [('E', None)]
    ]

    def generate_string(grammar):
        state = 0
        output = []
        while state is not None:
            index = np.random.randint(len(grammar[state]))
            production, state = grammar[state][index]
            if isinstance(production, list):
                production = generate_string(production)
            output.append(production)
        return ''.join(output)

    def generate_corrupted_string(grammar, chars='BEPSTVX'):
        good_string = generate_string(grammar)
        index = np.random.randint(len(good_string))
        good_char = good_string[index]
        bad_char = np.random.choice(sorted(set(chars) - set(good_char)))
        return good_string[:index] + bad_char + good_string[index + 1:]

    def string_to_one_hot_vectors(string, n_steps, chars='BEPSTVX'):
        char_to_index = {char: index for index, char in enumerate(chars)}
        output = np.zeros((n_steps, len(chars)), dtype=np.int32)
        for index, char in enumerate(string):
            output[index, char_to_index[char]] = 1.
        return output

    def generate_dataset(size):
        good_strings = [generate_string(grammar_lvl2)
                        for _ in range(size // 2)]
        bad_strings = [generate_corrupted_string(grammar_lvl2)
                       for _ in range(size - size // 2)]
        all_strings = good_strings + bad_strings
        n_steps = max([len(string) for string in all_strings])
        X = np.array([string_to_one_hot_vectors(string, n_steps)
                      for string in all_strings])
        seq_length = np.array([len(string) for string in all_strings])
        y = np.array([[1] for _ in range(len(good_strings))] +
                     [[0] for _ in range(len(bad_strings))])
        rnd_idx = np.random.permutation(size)
        return X[rnd_idx], seq_length[rnd_idx], y[rnd_idx]

    n_outputs = 1
    possible_chars = 'BEPSTVX'
    n_inputs = len(possible_chars)

    X = tf.placeholder(tf.float32, [None, None, n_inputs], name='X')
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
    y = tf.placeholder(tf.float32, [None, 1], name='y')

    gru_cell = rnn.GRUCell(num_units=30)
    outputs, states = tf.nn.dynamic_rnn(gru_cell, X, dtype=tf.float32, sequence_length=seq_length)
    logits = tf.layers.dense(states, n_outputs, name='logits')
    y_pred = tf.cast(tf.greater(logits, 0.), tf.float32, name='y_pred')
    y_proba = tf.nn.sigmoid(logits, name='y_proba')

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name='loss')
    optimizer = tf.train.MomentumOptimizer(learning_rate=.02, momentum=0.95, use_nesterov=True)
    training_op = optimizer.minimize(loss)
    correct = tf.equal(y_pred, y, name='correct')
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    X_train, length_train, y_train = generate_dataset(10000)
    X_val, length_val, y_val = generate_dataset(5000)
    n_epochs = 50
    batch_size = 50
    m = X_train.shape[0]
    loss_val = 0
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for batch_i in range(m // batch_size):
                l = batch_i * batch_size
                r = batch_i * batch_size + batch_size
                X_batch = X_train[l:r]
                length_batch = length_train[l:r]
                y_batch = y_train[l:r]
                loss_val, _ = sess.run([loss, training_op],
                                       feed_dict={X: X_batch, y: y_batch, seq_length: length_batch})
            acc_val_val = accuracy.eval(feed_dict={X: X_val, y: y_val, seq_length: length_val})
            print(
                '{:4d}  Train loss: {:.4f},  Validation accuracy: {:.2f}%'.
                    format(epoch, loss_val, 100 * acc_val_val)
            )
        saver.save(sess, '14_ex7_model')


exercise7()
