# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

learning_rate = 0.001
training_epochs = 15  # total training data을 한 번 train = 1 epoch
batch_size = 100 # 모든 데이터를 처리하지 않고 처리할 묶은 건수
# 모든데이터가 1000 이고 batch_size 100이면 1 epoch할려면 10번 반복작업이 실행됨
nb_classes = 10

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test_org) = mnist.load_data()

# normalizing data
x_train, x_test_normal = x_train / 255.0, x_test / 255.0

# change data shape
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test_normal.reshape(x_test_normal.shape[0], x_test_normal.shape[1] * x_test_normal.shape[2])

# change result to one-hot encoding
# in tf1, one_hot= True in read_data_sets("MNIST_data/", one_hot=True)
# took care of it, but here we need to manually convert them
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test_org, 10)

# # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
# array([0, 2, 1, 2, 0])
# `to_categorical` converts this into a matrix with as many columns as there are classes. The number of rows
#  stays the same. to_categorical(labels)
# array([[ 1.,  0.,  0.],
#        [ 0.,  0.,  1.],
#        [ 0.,  1.,  0.],
#        [ 0.,  0.,  1.],
#        [ 1.,  0.,  0.]], dtype=float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=10, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001), metrics=['accuracy'])
tf.model.summary()

print('fit---------------->')
history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(training_epochs)), history.history['loss'], label='loss')
plt.plot(list(np.arange(training_epochs)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.10.1.mnist_loss_accuracy_by_epoch_size")    
plt.show()

print('predict---------------->')
predictions = tf.model.predict(x_test)
print('Prediction: \n', predictions)
print(x_train)
score = tf.model.evaluate(x_train, y_train)
print('Accuracy(softmax, Adam): ', score[1])

# test 세트에 예측 결과 확인
pred = tf.model.predict_classes(x_test)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test_normal[i], cmap=plt.cm.binary)
    plt.xlabel(str(pred[i]) +','+ str(y_test_org[i]))
images.image.save_fig("tf2.10.1.mnist_test_images1_25")     
plt.show()

