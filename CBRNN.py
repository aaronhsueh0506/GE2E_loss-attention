#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
import time
import import_ipynb
import random
from utils import normalize, cal_er, _similarity, similarity, loss_cal
from tensorflow.contrib import rnn
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.device('/device:GPU:1')


# In[3]:


class model():
    def __init__(self, n_layer, state_size, lr, em_size, batch_size, n_file, dropout_ratio=False):
        self.mode = None

        self.n_layer = n_layer
        self.hidden = state_size
        self.lr = lr
        self.em_size = em_size
        self.dropout_ratio = dropout_ratio
        self.M = n_file
        self.N = batch_size
        self.proj = 64
        self.P = self.proj

        self.graph = tf.Graph()
        self.build()
        self.sess=tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def build(self):
        with self.graph.as_default():
            ### input ###
            self.X = tf.placeholder(tf.float32, [None, 99, 40], name='X')  # [batch, frames, feature]
            self.x = tf.reshape(self.X, [-1, 99, 40, 1])

            w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
            b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Trainable parameters
            initializer = tf.random_normal_initializer(stddev=0.1)
            w_omega = tf.get_variable(name="w_omega", shape=[self.em_size, self.hidden], initializer=initializer)
            b_omega = tf.get_variable(name="b_omega", shape=[self.hidden], initializer=initializer)
            u_omega = tf.get_variable(name="u_omega", shape=[self.hidden], initializer=initializer)

            ### CNN ###
            conv1 = tf.layers.conv2d(inputs=self.x,
                                     filters=16,
                                     kernel_size=[5,5],
                                     padding='same',
                                     # activation=tf.nn.relu,
                                     name='conv1')
            bn1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
            pool1 = tf.layers.max_pooling2d(inputs=bn1,
                                            pool_size=[2,2],
                                            strides=[1,2],
                                            name='pool1')

            ### LN-LSTM ###
#             cells = [tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.hidden),
#                                                             output_size=self.em_size) for _ in range(self.n_layer)]

            # cells = [tf.contrib.rnn.LSTMCell(num_units = self.hidden, num_proj= self.em_size) for _ in range(self.n_layer)]
            # lstm = tf.contrib.rnn.MultiRNNCell(cells)    # define lstm op and variables
            # outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=self.X, dtype=tf.float32)   # for TI-VS must use dynamic rnn

            for i in range(n_layer):
                cell = tf.contrib.rnn.LSTMCell(num_units = self.hidden, num_proj= self.em_size)
                cells_fw.append(cell)
                cell = tf.contrib.rnn.LSTMCell(num_units = self.hidden, num_proj= self.em_size)
                cells_bw.append(cell)
            fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
            bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, self.X)

            embedded = attention(outputs, w_omega, b_omega, u_omega)
            self.embedded = normalize(embedded)                    # normalize
            print(self.embedded)

            ### similarity martix ###
            sim_matrix = similarity(self.embedded, w, b, self.M, self.N, self.em_size)

            ### calculate SV loss ###
            self.loss = loss_cal(sim_matrix, self.M, self.N, type='contrast')

            trainable_vars= tf.trainable_variables()                # get variable list
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            grads, vars= zip(*optimizer.compute_gradients(self.loss))
            grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)
            grads_rescale= [0.01*grad for grad in grads_clip[:2]] + grads_clip[2:]
            self.train_op = optimizer.apply_gradients(zip(grads_rescale, vars))

            self.init_op = tf.global_variables_initializer()

    def fit(self, x, iters):
        self.mode = True
        feed_dict={self.X: x}
        self.sess.run(self.train_op, feed_dict=feed_dict)
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def extract(self, x):
        self.mode=False
        feed_dict={self.X: x}
        predict = self.sess.run(self.embedded, feed_dict=feed_dict)
        return predict

    def _saver(self, epoch):
        self.mode=False
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=0)
            saver.save(self.sess, "./save_model/Check_Point/model.ckpt", global_step=epoch//25)

    def _restore(self, index):
        self.mode=False
        with self.graph.as_default():
            saver = tf.train.Saver(var_list=tf.global_variables())
            model = "./save_model/Check_Point/model.ckpt-" + str(index)
            saver.restore(self.sess, model)


# In[4]:


train_data = np.load('/home/mingyu/array/variable/MFSC/truth_train.npy')


# In[5]:


def generator(number, _data):
#     train_data = np.load('/home/mingyu/Speaker_id/Model/CBRNN/train.npy')
    num = len(number)
    batch = np.empty((0,99,40))
    for i in range(num):
        batch = np.concatenate((batch, _data[number[i]]), axis=0)
    return batch


# In[6]:


epoch = 10000
batch_size = 64
state_size = 128
n_layer = 3
lr = 10**-2
em_size = 64
n_file = 5
dropout_ratio = 0
n_train = 1186

m = model(n_layer,
         state_size,
         lr,
         em_size,
         batch_size,
         n_file,
         dropout_ratio=dropout_ratio)


# In[ ]:


print('start!')
for epochs in range(epoch):
    all_loss = 0
    train_index = [i for i in range(n_train)]
    random.shuffle(train_index)

    while len(train_index)>batch_size:
        batch_index = [train_index.pop() for _ in range(min(batch_size, len(train_index)))]

        # generate
        x = generator(batch_index, train_data)

        # training
        loss = m.fit(x=x, iters=epochs+1)
        all_loss += loss

        print('[%3d/%3d] ,loss: %5.2f'%(n_train-len(train_index), n_train, loss),end = '\r')

    print('epoch:%3d'%(epochs+1),', loss: %5.2f'%(all_loss),end = '\n')

    if epochs % 25 == 0: m._saver(epochs)

print('complete!')


# In[ ]:


# bi-LSTM with BN, sv for sgd
