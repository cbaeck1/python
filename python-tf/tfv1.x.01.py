# 2.0 에서 작동하지 않음

import tensorflow as tf

xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

H = W*X + b 
cost = tf.reduce_mean(tf.square(H - y))
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(5001):
  sess.run(train, feed_dict={X: xData, y: yData})
  if i % 500 == 0:
    print(i, sess.run(cost, feed_dict={X: xData, y: yData}), sess.run(W), sess.run(b))

print(sess.run(H, feed_dict={X: [8]}))    
