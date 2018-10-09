#! /usr/bin/python3

import tensorflow as tf

input = tf.constant([[1,2,3],[4,5,6],[8,9,7]])
print(input.shape)
slice_input = tf.slice(input, [0, 0], [3, -1])

with tf.Session() as sess:
	print(sess.run(slice_input))