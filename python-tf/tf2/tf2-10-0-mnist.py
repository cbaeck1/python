# Lab 7 Learning rate and Evaluation
import tensorflow as tf
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

# 훈련 세트에 있는 첫 번째 이미지를 보면 픽셀 값의 범위가 0~255 사이
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
images.image.save_fig("tf2.10.0.mnist_train_images")     
plt.show()

# normalizing data
x_train, x_test_normal = x_train / 255.0, x_test / 255.0

# 훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름을 출력
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
images.image.save_fig("tf2.10.0.mnist_train_images1_25")     
plt.show()

