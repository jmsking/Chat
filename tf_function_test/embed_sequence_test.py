#! /usr/bin/python3

import tensorflow as tf

inputs = [[1,2,3,4,5], [6,1,8,5,10], [10,3,1,9,7]]
vocab_size = 15
embed_dim = 8

embed_inputs = tf.contrib.layers.embed_sequence(inputs, vocab_size, embed_dim)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer()) 
  print(sess.run(embed_inputs).shape)
  print(sess.run(embed_inputs))