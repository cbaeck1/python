import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == stochastic gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
iterations = 50
history = tf.model.fit(x_train, y_train, epochs=iterations, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(iterations)), history.history['loss'], label='loss')
# plt.plot(list(np.arange(iterations)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.2.1.lr_mse_SGD_loss_by_epoch_size")    
plt.show()

# predict() returns predicted value
# np.array([4, 5])
x_test = [5, 6]
y_predict = tf.model.predict(x_test)
print(y_predict, y_predict.shape)

x_train = x_train + x_test
y_train = y_train + y_predict.reshape(2, order='C').tolist()
# y_train = y_train + y_predict.tolist() # [0] +  y_predict.tolist()[1]
print(x_train, y_train)

# Plot predictions
plt.plot(x_train)
plt.plot(y_train)
# plt.xlabel("Origin")
# plt.ylabel("Predict")
images.image.save_fig("tf2.2.1.lr_mse_SGD_compare")  
plt.show()


