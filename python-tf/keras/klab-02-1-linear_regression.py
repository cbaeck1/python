import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

model = Sequential()
model.add(Dense(1, input_dim=1))

sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)

# prints summary of the model to the terminal
model.summary()

iterations = 200
history = model.fit(x_train, y_train, epochs=iterations, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(iterations)), history.history['loss'], label='loss')
#plt.plot(list(np.arange(iterations)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("klab.2.1.lr_mse_SGD_loss_by_epoch_size")    
plt.show()

x_test = [5, 6]
y_predict = model.predict(x_test)
print(y_predict)

x_train = x_train + x_test
y_train = y_train + y_predict.reshape(2, order='C').tolist()
print(x_train, y_train)

# Plot predictions
plt.plot(x_train)
plt.plot(y_train)
# plt.xlabel("Origin")
# plt.ylabel("Predict")
images.image.save_fig("klab.2.1.lr_mse_SGD_compare")  
plt.show()
