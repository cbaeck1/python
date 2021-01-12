import tensorflow as tf 
print(tf.__version__)

hello = tf.constant('hello') 
# 텐서플로우 버전 2 이상에서는 Session을 정의하고 run 해주는 과정이 생략된다. 
# sess = tf.Session() 
# print(sess.run(hello))
tf.print(hello)

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
tf.print(node1,node2)


