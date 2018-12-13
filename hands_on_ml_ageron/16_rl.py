import os

import gym
import numpy as np
import tensorflow as tf


def open_ai_demo():
    reward_list = []
    env = gym.make('CartPole-v0')

    n_max_steps = 1000

    obs = env.reset()
    for step in range(n_max_steps):
        env.render()

        position, velocity, angle, angular_velocity = obs
        if angle < 0:
            action = 0
        else:
            action = 1

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        if done:
            break
    env.close()
    print('Steps made: {}'.format(len(reward_list)))


def gradient_policy_demo():
    def discount_rewards(rewards, discount_factor):
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_factor
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(all_rewards, discount_rate):
        all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    n_inputs = 4
    n_hidden = n_inputs
    n_outputs = 1
    lr = .01

    initializer = tf.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs, name='logits')
    # prbty of action 0 (left)
    outputs = tf.nn.sigmoid(logits)
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    y = 1. - tf.to_float(action)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_games_per_update = 10
    n_max_steps = 1000
    n_iterations = 250
    save_iterations = 10
    discount_factor = 0.95
    model_filename = './16_GP_demo.ckpt'

    try:
        with tf.Session() as sess:
            saver.restore(sess, model_filename)
            is_trained = True
    except ValueError:
        is_trained = False

    if not is_trained:
        env = gym.make('CartPole-v0')
        with tf.Session() as sess:
            init.run()
            for iteration in range(n_iterations):
                print('\rIteration: {}'.format(iteration), end='')
                all_rewards = []
                all_gradients = []
                for game in range(n_games_per_update):
                    current_rewards = []
                    current_gradients = []
                    obs = env.reset()
                    for step in range(n_max_steps):
                        action_val, gradients_val = sess.run([action, gradients],
                                                             feed_dict={X: obs.reshape(1, n_inputs)})
                        obs, reward, done, info = env.step(action_val[0][0])
                        current_rewards.append(reward)
                        current_gradients.append(gradients_val)
                        if done:
                            break
                    all_rewards.append(current_rewards)
                    all_gradients.append(current_gradients)
                all_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
                feed_dict = {}
                for var_idx, gradient_placeholder in enumerate(gradient_placeholders):
                    mean_gradients = np.mean([reward * all_gradients[game_idx][step][var_idx]
                                              for game_idx, rewards in enumerate(all_rewards)
                                              for step, reward in enumerate(rewards)], axis=0)
                    feed_dict[gradient_placeholder] = mean_gradients
                sess.run(training_op, feed_dict=feed_dict)
                if iteration % save_iterations == 0:
                    saver.save(sess, model_filename)
        env.close()

    env = gym.make('CartPole-v0')
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_filename)
        for step in range(n_max_steps):
            env.render()
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
        print('real steps taken: {}'.format(step))
    env.close()


def q_value_demo():
    nan = np.nan  # represents impossible actions
    T = np.array([  # shape=[s, a, s']
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
        [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
    ])
    R = np.array([  # shape=[s, a, s']
        [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
        [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
    ])
    possible_actions = [[0, 1, 2], [0, 2], [1]]

    Q = np.full((3, 3), -np.inf)  # -inf for impossible actions
    for state, actions in enumerate(possible_actions):
        Q[state, actions] = 0.0  # Initial value = 0.0, for all possible actions

    discount_rate = 0.9
    n_iterations = 100
    for iteration in range(n_iterations):
        Q_prev = Q.copy()
        for s in range(3):
            for a in possible_actions[s]:
                Q[s, a] = np.sum([
                    T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prev[sp]))
                    for sp in range(3)
                ])
    print(Q)


def qlearning_demo():
    class MDPEnvironment(object):
        def __init__(self, start_state=0):
            self.start_state = start_state
            self.reset()

            self.transition_probabilities = [
                # in s0, if action a0 then proba 0.7 to state s0 and 0.3 to state s1, etc.
                [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
                [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
                [None, [0.8, 0.1, 0.1], None],
            ]

            self.rewards = [
                [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
                [[0, 0, 0], [+40, 0, 0], [0, 0, 0]],
            ]

            self.possible_actions = []
            for from_state_list in self.transition_probabilities:
                possible_states = []
                for idx, to_states in enumerate(from_state_list):
                    if to_states is not None:
                        possible_states.append(idx)
                self.possible_actions.append(possible_states)
            self.state_num = len(self.transition_probabilities)

        def reset(self):
            self.total_rewards = 0
            self.state = self.start_state

        def step(self, action):
            next_state = np.random.choice(
                range(self.state_num),
                p=self.transition_probabilities[self.state][action]
            )
            reward = self.rewards[self.state][action][next_state]
            self.state = next_state
            self.total_rewards += reward
            return self.state, reward

    def run_episode(policy, n_steps, display=True):
        env = MDPEnvironment()
        if display:
            print('States (+rewards):', end=' ')
        for step in range(n_steps):
            if display:
                if step == 10:
                    print('...', end=' ')
                elif step < 10:
                    print(env.state, end=' ')
            action = policy(env.state)
            state, reward = env.step(action)
            if display and step < 10:
                if reward:
                    print('({})'.format(reward), end=' ')
        if display:
            print('Total rewards =', env.total_rewards)
        return env.total_rewards

    env = MDPEnvironment()

    def policy_fire(state):
        return [0, 2, 1][state]

    def policy_random(state):
        return np.random.choice(env.possible_actions[state])

    def policy_safe(state):
        return [0, 0, 1][state]

    for policy in (policy_fire, policy_random, policy_safe):
        all_totals = []
        print(policy.__name__)
        for episode in range(1000):
            all_totals.append(run_episode(policy, n_steps=100, display=(episode < 5)))
        print('Summary: mean={:.1f}, std={:1f}, min={}, max={}'.format(np.mean(all_totals), np.std(all_totals),
                                                                       np.min(all_totals), np.max(all_totals)))

    n_states = 3
    n_actions = 3
    n_steps = 20000
    alpha = 0.01
    gamma = 0.99
    exploration_policy = policy_random
    q_values = np.full((n_states, n_actions), -np.inf)
    for state, actions in enumerate(env.possible_actions):
        q_values[state][actions] = 0

    for step in range(n_steps):
        action = exploration_policy(env.state)
        state = env.state
        next_state, reward = env.step(action)
        next_value = np.max(q_values[next_state])  # greedy policy
        q_values[state, action] = (1 - alpha) * q_values[state, action] + alpha * (reward + gamma * next_value)

    def optimal_policy(state):
        return np.argmax(q_values[state])

    all_totals = []
    for episode in range(1000):
        all_totals.append(run_episode(optimal_policy, n_steps=100, display=(episode < 5)))
    print('Summary: mean={:.1f}, std={:1f}, min={}, max={}'.format(np.mean(all_totals), np.std(all_totals),
                                                                   np.min(all_totals), np.max(all_totals)))
    print()


def pacman_demo():
    class ReplayMemory:
        def __init__(self, maxlen):
            self.maxlen = maxlen
            self.buf = np.empty(shape=maxlen, dtype=np.object)
            self.index = 0
            self.length = 0

        def append(self, data):
            self.buf[self.index] = data
            self.length = min(self.length + 1, self.maxlen)
            self.index = (self.index + 1) % self.maxlen

        def sample(self, batch_size, with_replacement=True):
            if with_replacement:
                indices = np.random.randint(self.length, size=batch_size)  # faster
            else:
                indices = np.random.permutation(self.length)[:batch_size]
            return self.buf[indices]

    def sample_memories(batch_size):
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for memory in replay_memory.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def preprocess_observation(obs):
        mspacman_color = np.array([210, 164, 74]).mean()
        img = obs[1:176:2, ::2]  # crop and downsize
        img = img.mean(axis=2)  # to greyscale
        img[img == mspacman_color] = 0  # improve contrast
        img = (img - 128) / 128 - 1  # normalize from -1. to 1.
        return img.reshape(88, 80, 1)

    def epsilon_greedy(q_values, step):
        epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(n_outputs)  # random action
        else:
            return np.argmax(q_values)  # optimal action

    def q_network(X_state, name):
        prev_layer = X_state / 128.0  # scale pixel intensities to the [-1.0, 1.0] range.
        with tf.variable_scope(name) as scope:
            for n_maps, kernel_size, strides, padding, activation in zip(
                    conv_n_maps, conv_kernel_sizes, conv_strides,
                    conv_paddings, conv_activation):
                prev_layer = tf.layers.conv2d(
                    prev_layer, filters=n_maps, kernel_size=kernel_size,
                    strides=strides, padding=padding, activation=activation,
                    kernel_initializer=initializer)
            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
            hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                     activation=hidden_activation,
                                     kernel_initializer=initializer)
            outputs = tf.layers.dense(hidden, n_outputs,
                                      kernel_initializer=initializer)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        return outputs, trainable_vars_by_name

    env = gym.make('MsPacman-v0')
    env.reset()
    input_height = 88
    input_width = 80
    input_channels = 1
    conv_n_maps = [32, 64, 64]
    conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
    conv_strides = [4, 2, 1]
    conv_paddings = ['SAME'] * 3
    conv_activation = [tf.nn.relu] * 3
    n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
    n_hidden = 512
    hidden_activation = tf.nn.relu
    n_outputs = env.action_space.n  # 9 discrete actions are available
    initializer = tf.variance_scaling_initializer()

    X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
    online_q_values, online_vars = q_network(X_state, name='q_networks/online')
    target_q_values, target_vars = q_network(X_state, name='q_networks/target')

    copy_ops = [target_var.assign(online_vars[var_name])
                for var_name, target_var in target_vars.items()]
    copy_online_to_target = tf.group(*copy_ops)

    learning_rate = 0.001
    momentum = 0.95
    with tf.variable_scope('train'):
        X_action = tf.placeholder(tf.int32, shape=[None])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                                axis=1, keepdims=True)
        error = tf.abs(y - q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        training_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    replay_memory_size = 500000
    replay_memory = ReplayMemory(replay_memory_size)
    eps_min = 0.1
    eps_max = 1.0
    eps_decay_steps = 2000000

    n_steps = 4000000  # total number of training steps
    training_start = 10000  # start training after 10,000 game iterations
    training_interval = 4  # run a training step every 4 game iterations
    save_steps = 1000  # save the model every 1,000 training steps
    copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
    discount_rate = 0.99
    skip_start = 90  # Skip the start of every game (it's just waiting time).
    batch_size = 50
    iteration = 0  # game iterations
    checkpoint_path = './15_dqn.ckpt'
    done = True  # env needs to be reset
    loss_val = np.infty
    game_length = 0
    total_max_q = 0
    mean_max_q = 0.0

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path + '.index'):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()
        while True:
            step = global_step.eval()
            if step >= n_steps:
                break
            iteration += 1
            print('\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   '.format(
                iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end='')
            if done:  # game over, start again
                obs = env.reset()
                for skip in range(skip_start):  # skip the start of each game
                    obs, reward, done, info = env.step(0)
                state = preprocess_observation(obs)

            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_values, step)

            # Online DQN plays
            obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(obs)

            # Let's memorize what happened
            replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress (not shown in the book)
            total_max_q += q_values.max()
            game_length += 1
            if done:
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0

            if iteration < training_start or iteration % training_interval != 0:
                continue  # only train after warmup period and at regular intervals

            # Sample memories and use the target DQN to produce the target Q-Value
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                sample_memories(batch_size))
            next_q_values = target_q_values.eval(
                feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            _, loss_val = sess.run([training_op, loss], feed_dict={
                X_state: X_state_val, X_action: X_action_val, y: y_val})

            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)


qlearning_demo()
