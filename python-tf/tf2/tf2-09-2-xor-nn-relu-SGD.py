# Lab 9 XOR
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# layer를 하나 더 추가
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=2, units=2, use_bias=True))
tf.model.add(tf.keras.layers.Activation('relu'))
tf.model.add(tf.keras.layers.Dense(input_dim=2, units=1))
tf.model.add(tf.keras.layers.Activation('sigmoid'))
tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(lr=0.1),  metrics=['accuracy'])
tf.model.summary()

epoch_size = 1000
history = tf.model.fit(x_data, y_data, epochs=epoch_size, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(epoch_size)), history.history['loss'], label='loss')
plt.plot(list(np.arange(epoch_size)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.9.2.xor_nn_relu_SGD_loss_accuracy_by_epoch_size")    
plt.show()

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy(nn,relu,SGD): ', score[1])
