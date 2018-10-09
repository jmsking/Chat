#! /usr/bin/python3

import tensorflow as tf

scale = tf.constant(1.0)
rand_val = tf.random_normal((5,))
#x = tf.random_normal((3,5))
tmp = scale*rand_val
#y = x + tmp
with tf.Session() as sess:
  print(sess.run(rand_val))
  #print(sess.run(x))
  print(sess.run(rand_val))
  #print(sess.run(y))