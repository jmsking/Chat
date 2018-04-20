#! /usr/bin/python3
import tensorflow as tf
import numpy as np
# 创建输入数据
X = np.random.randn(2, 10, 8)

# 第二个example长度为6
X[1,6:] = 0
X_lengths = [10, 6]

cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print(result[0]["outputs"].shape)
print(result[0]['last_states'][0].shape)
print(result[0]['last_states'][1].shape)