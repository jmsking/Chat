#! /usr/bin/python3

import tensorflow as tf
import numpy as np

#input = tf.constant([[1,2,3,4],[5,6,7,8]])
input = np.array([[1,2,3,4],[5,6,7,8]])
output = tf.identity(input)
with tf.Session() as sess:
	print(sess.run(output))