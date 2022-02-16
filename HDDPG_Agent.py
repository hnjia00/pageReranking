import tensorflow as tf
import numpy as np
from open_data.config import *


###############################  DDPG  ####################################

class meta_controller_DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + rnn_hidden_dim), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self._batch_size = tf.placeholder(shape=[1], dtype=tf.int32)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.rnn_in_state = np.zeros([1, rnn_hidden_dim])
        self.rnn_out_state = np.zeros([1, rnn_hidden_dim])
        gru_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_dim)
        self.init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        self.a, self.last_state = self._build_a(self.S, gru_cell, self.init_state)

        self.critic_rnn_in_state = np.zeros([1, rnn_hidden_dim])
        self.critic_rnn_out_state = np.zeros([1, rnn_hidden_dim])
        critic_gru_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_dim)
        self.critic_init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        q, _ = self._build_c(self.S, self.a, critic_gru_cell, self.critic_init_state)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        target_gru_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_dim)
        self.target_init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        a_, _ = self._build_a(self.S_, target_gru_cell, self.target_init_state, reuse=True, custom_getter=ema_getter)  # replaced target parameters

        critic_target_gru_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_dim)
        self.critic_target_init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        q_, _ = self._build_c(self.S_, a_, critic_target_gru_cell, self.critic_init_state, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        # tf.summary.scalar("critic_q", q)
        #
        # self.writer = tf.summary.FileWriter('./data/graph', self.sess.graph)
        # self.merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        self.rnn_in_state = self.rnn_out_state
        a, self.rnn_out_state = self.sess.run([self.a, self.last_state], {self.S: s[np.newaxis, :], self.init_state: self.rnn_in_state, self._batch_size :[1]})
        return a

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]
        br =  bt[:,  self.s_dim + self.a_dim:  self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1: self.s_dim + self.a_dim + 1 + self.s_dim]
        brnn = bt[:, self.s_dim + self.a_dim + 1 + self.s_dim: ]

        self.sess.run(self.atrain, {self.S: bs, self.init_state: brnn, self._batch_size: [BATCH_SIZE]})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self._batch_size: [BATCH_SIZE],self.critic_init_state: brnn})

        # result = self.sess.run(self.merged, {self.S: bs, self.init_state: brnn, self._batch_size: [BATCH_SIZE]})
        # self.writer.add_summary(result, self.pointer)

    def store_transition(self, s, a, r, s_):
        rnn_state = np.squeeze(self.rnn_in_state)
        transition = np.hstack((s, a, [r], s_, rnn_state))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, rnn_cell, init_state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            mlp = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)

            rnn_in = tf.expand_dims(mlp, axis=0)
            rnn_out, last_state = tf.nn.dynamic_rnn(rnn_cell, rnn_in, initial_state=init_state,
                                                    dtype=tf.float32, time_major=True)
            rnn = tf.squeeze(rnn_out, axis=[0])

            a = tf.layers.dense(rnn, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return tf.clip_by_value(a, 0, 100000), last_state

    def _build_c(self, s, a, rnn_cell, init_state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 32
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            mlp = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            rnn_in = tf.expand_dims(mlp, axis=0)
            rnn_out, last_state = tf.nn.dynamic_rnn(rnn_cell, rnn_in, initial_state=init_state,
                                                    dtype=tf.float32, time_major=True)
            rnn = tf.squeeze(rnn_out, axis=[0])

            q = tf.layers.dense(rnn, 1, trainable=trainable)  # Q(s,a)
            return q, last_state


class controller_DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((c_MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + c_rnn_hidden_dim), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self._batch_size = tf.placeholder(shape=[1], dtype=tf.int32)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 'c_s')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 'c_s_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'c_r')

        self.rnn_in_state = np.zeros([1, c_rnn_hidden_dim])
        self.rnn_out_state = np.zeros([1, c_rnn_hidden_dim])
        gru_cell = tf.nn.rnn_cell.GRUCell(c_rnn_hidden_dim)
        self.init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        self.a, self.last_state = self._build_a(self.S, gru_cell, self.init_state)

        self.critic_rnn_in_state = np.zeros([1, c_rnn_hidden_dim])
        self.critic_rnn_out_state = np.zeros([1, c_rnn_hidden_dim])
        critic_gru_cell = tf.nn.rnn_cell.GRUCell(c_rnn_hidden_dim)
        self.critic_init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        q, _ = self._build_c(self.S, self.a, critic_gru_cell, self.critic_init_state)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c_Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c_Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - c_TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        target_gru_cell = tf.nn.rnn_cell.GRUCell(c_rnn_hidden_dim)
        self.target_init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        a_, _ = self._build_a(self.S_, target_gru_cell, self.target_init_state, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        critic_target_gru_cell = tf.nn.rnn_cell.GRUCell(c_rnn_hidden_dim)
        self.critic_target_init_state = gru_cell.zero_state(self._batch_size, tf.float32)
        q_, _ = self._build_c(self.S_, a_, critic_target_gru_cell, self.critic_init_state, reuse=True,
                              custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(c_LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(c_LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        self.rnn_in_state = self.rnn_out_state
        a, self.rnn_out_state = self.sess.run([self.a, self.last_state],
                                              {self.S: s[np.newaxis, :], self.init_state: self.rnn_in_state,
                                               self._batch_size: [1]})
        return a

    def learn(self):
        indices = np.random.choice(c_MEMORY_CAPACITY, size=c_BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]
        br =  bt[:,  self.s_dim + self.a_dim:  self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1: self.s_dim + self.a_dim + 1 + self.s_dim]
        brnn = bt[:, self.s_dim + self.a_dim + 1 + self.s_dim: ]

        self.sess.run(self.atrain, {self.S: bs, self.init_state: brnn, self._batch_size: [c_BATCH_SIZE]})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self._batch_size: [c_BATCH_SIZE],self.critic_init_state: brnn})

    def store_transition(self, s, a, r, s_):
        rnn_state = np.squeeze(self.rnn_in_state) # TODO: 保存cirtic的rnn state转移
        transition = np.hstack((s, a, [r], s_, rnn_state))
        index = self.pointer % c_MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, rnn_cell, init_state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('c_Actor', reuse=reuse, custom_getter=custom_getter):
            mlp = tf.layers.dense(s, 64, activation=tf.nn.relu, name='c_l1', trainable=trainable)

            rnn_in = tf.expand_dims(mlp, axis=0)
            rnn_out, last_state = tf.nn.dynamic_rnn(rnn_cell, rnn_in, initial_state=init_state,
                                                    dtype=tf.float32, time_major=True)
            rnn = tf.squeeze(rnn_out, axis=[0])

            a = tf.layers.dense(rnn, self.a_dim, activation=tf.nn.tanh, name='c_a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return tf.clip_by_value(a, 0, 100000), last_state

    def _build_c(self, s, a, rnn_cell, init_state, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('c_Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            w1_s = tf.get_variable('c_w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('c_w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('c_b1', [1, n_l1], trainable=trainable)
            mlp = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            rnn_in = tf.expand_dims(mlp, axis=0)
            rnn_out, last_state = tf.nn.dynamic_rnn(rnn_cell, rnn_in, initial_state=init_state,
                                                    dtype=tf.float32, time_major=True)
            rnn = tf.squeeze(rnn_out, axis=[0])

            q = tf.layers.dense(rnn, 1, trainable=trainable)  # Q(s,a)
            return q, last_state

    def reset(self):
        self.rnn_in_state = self.rnn_out_state = np.zeros([1, c_rnn_hidden_dim])