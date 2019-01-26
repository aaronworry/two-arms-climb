import tensorflow as tf
import numpy as np

MEMORY_CAPACITY = 30000
BATCH_SIZE = 32
GAMMA = 0.9
LR = 0.001
EPSILONMAX = 0.9


class QL():
    def __init__(self, a_dim, s_dim):
        self.sess = tf.Session()
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 1 + 1), dtype=np.float32)
        self.memory_full = False
        self.pointer = 0
        self.learnStep = 0
        self.epsilon = 0
        self._buildNetwork()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess.run(tf.global_variables_initializer())
        self.Cost = []

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: s[None,:]})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.a_dim)
        return action

    def learn(self):
        if self.learnStep % 300 == 0:
            self.sess.run(self.replace_target_op)
        if self.pointer > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.pointer, size=BATCH_SIZE)
        batchM = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
            feed_dict={
                self.s_: batchM[:, -self.s_dim:],  # fixed params
                self.s: batchM[:, :self.s_dim],  # newest params
            })

        q_target = q_eval.copy()
        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
        eval_act_index = batchM[:, self.s_dim].astype(int)
        reward = batchM[:, self.s_dim + 1]

        q_target[batch_index, eval_act_index] = reward + GAMMA * np.max(q_next, axis=1)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batchM[:, :self.s_dim],
                                                self.q_target: q_target})
        self.Cost.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + 0.001 if self.epsilon < EPSILONMAX else EPSILONMAX
        self.learnStep += 1


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'pointer'):
            self.pointer = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


    def _buildNetwork(self):
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], name='s_')
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name = 's')
        self.q_target = tf.placeholder(tf.float32, [None, self.a_dim], name='Q_target')
        with tf.variable_scope('eval'):
            nl1 = 300
            w1 = tf.get_variable('w1', [self.s_dim, nl1])
            b1 = tf.get_variable('b1', [1, nl1])
            fc1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            w2 = tf.get_variable('w2', [nl1, self.a_dim])
            b2 = tf.get_variable('b2', [1, self.a_dim])
            self.q_eval = tf.matmul(fc1, w2) + b2
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(LR).minimize(self.loss)
        with tf.variable_scope('target'):
            nl1 = 300
            w1 = tf.get_variable('w1', [self.s_dim, nl1])
            b1 = tf.get_variable('b1', [1, nl1])
            fc1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            w2 = tf.get_variable('w2', [nl1, self.a_dim])
            b2 = tf.get_variable('b2', [1, self.a_dim])
            self.q_next = tf.matmul(fc1, w2) + b2


    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'params/Q-learning/params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'params/Q-learning/params')