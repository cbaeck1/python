'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('data/data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

# train/test split
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, :]
        y = time_series[i + seq_length, [-1]]  # Next close price
        print(x, "->", y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)  # (505, 7, 5)
print(trainY.shape)  # (505, 1)
print(testX.shape)  # (220, 7, 5)
print(testY.shape)  # (220, 1)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=1, input_shape=(seq_length, data_dim)))
tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
tf.model.summary()

tf.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
history = tf.model.fit(trainX, trainY, epochs=iterations, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(iterations)), history.history['loss'], label='loss')
# plt.plot(list(np.arange(iterations)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.12.5.stock_rnn_tanh_Adam_loss_by_epoch_size")    
plt.show()

# Test step
test_predict = tf.model.predict(testX)
print('Prediction: \n', test_predict, test_predict.shape)

score = tf.model.evaluate(testX, testY)
# print('Score: \n', score)
print('Accuracy(rnn,tanh,Adam): ', score)

print(testY.shape, test_predict.shape)
# print(testY)
# print(test_predict)
# Plot predictions
plt.grid(True)
plt.plot(testY, label='testY')
plt.plot(test_predict, label='test_predict')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend('best')
images.image.save_fig("tf2.12.5.stock_rnn_mse_Adam_compare")  
plt.show()
