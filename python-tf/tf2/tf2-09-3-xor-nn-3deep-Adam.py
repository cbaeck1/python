# Lab 9 XOR
# 9-3 deep 
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'))
# 3 deep
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))

tf.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# SGD not working very well due to vanishing gradient problem, switched to Adam for now
# or you may use activation='relu', study chapter 10 to know more on vanishing gradient problem.
tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.1), metrics=['accuracy'])
tf.model.summary()

epoch_size = 1000
history = tf.model.fit(x_data, y_data, epochs=epoch_size, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(epoch_size)), history.history['loss'], label='loss')
plt.plot(list(np.arange(epoch_size)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.9.3.xor_nn_3deep_Adam_loss_accuracy_by_epoch_size")    
plt.show()

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy(3deep,sigmoid,Aadm): ', score[1])
