# Lab 9 XOR
# But it doesn't work (왜) 직선 하나로 XOR를 구별할 수 없다.
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

print(x_data.shape, y_data.shape)
print(x_data)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=2, units=1, activation='sigmoid'))
tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01),  metrics=['accuracy'])
tf.model.summary()

epoch_size = 10000
history = tf.model.fit(x_data, y_data, epochs=epoch_size, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(epoch_size)), history.history['loss'], label='loss')
plt.plot(list(np.arange(epoch_size)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.9.2.xor_loss_accuracy_by_epoch_size")    
plt.show()

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])

