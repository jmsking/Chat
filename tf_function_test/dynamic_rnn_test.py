#! /usr/bin/python3
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
# 创建输入数据
#X1 = np.random.randn(1, 1, 3)
X = tf.random_normal(shape=[5,1,1], dtype=tf.float32)
#X = tf.reshape(X, [-1, 1, 10])

# 第二个example长度为6
#X[1,6:] = 0
#X_lengths = [10, 6]

'''cell = rnn.BasicLSTMCell(num_units=64)
multi_cell = rnn.MultiRNNCell([cell for _ in range(2)])
#initial_state = multi_cell.zero_state(2,dtype=tf.float64)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=multi_cell,
    #sequence_length=X_lengths,
    inputs=X1,
	dtype=tf.float64)'''
	
	
# create 3 LSTMCells
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=3, state_is_tuple=True)
rnn_layers = [tf.nn.rnn_cell.BasicLSTMCell(size) for size in [256, 128, 64]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
print(multi_rnn_cell.state_size)
print(multi_rnn_cell.output_size)
# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
outputs, last_states = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=X,
                                   dtype=tf.float32)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print(result[0]["outputs"].shape)
print(result[0]['last_states'][0][0].shape)
print(result[0]['last_states'][0][1].shape)
print(result[0]['last_states'][1][0].shape)
print(result[0]['last_states'][1][1].shape)
print(result[0]['last_states'][2][0].shape)
print(result[0]['last_states'][2][1].shape)