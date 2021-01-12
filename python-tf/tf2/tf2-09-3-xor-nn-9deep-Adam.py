# Lab 9 XOR
# 9-3 9 layer deep 
import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# SGD not working very well due to vanishing gradient problem, switched to Adam for now
# or you may use activation='relu', study chapter 10 to know more on vanishing gradient problem.
tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.1), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=1000, verbose=0)

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy(9deep,sigmoid,Adam): ', score[1])
