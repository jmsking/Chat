#! /usr/bin/python3

import tensorflow as tf

#features = tf.truncated_normal(shape=[1], stddev=1)
features = tf.constant([[1,2,3,-0.2],[-0.1, 2, -3, -100]])

out = tf.nn.relu(features, name='relu')

with tf.Session() as sess:
	print('features: ', sess.run(features))
	print('out: ', sess.run(out))