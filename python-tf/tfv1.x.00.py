# 


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()
# sess = tf.Session()
sess.run(hello)
a = tf.constant(10)
b = tf.constant(32)
sess.run(a + b)

