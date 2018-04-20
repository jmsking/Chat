#! /usr/bin/python3

import tensorflow as tf

init_float = tf.random_uniform_initializer(0, 1, 5000, tf.float32)
init_int = tf.random_uniform_initializer(0, 100, 5000, tf.int32)

with tf.Session() as sess:
  var_float = tf.get_variable("var_float", shape=[2,3], initializer=init_float)
  var_int = tf.get_variable("var_int", shape = [2,3], initializer=init_int)
  #var_int.initializer.run()
  #var_float.initializer.run()
  sess.run(tf.global_variables_initializer())
  print(var_int.eval())
  print(var_float.eval())