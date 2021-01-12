import keras
import matplotlib.pyplot as plt
import image

# 뉴스 기사 분류: 다중 분류 문제
# 로이터 뉴스를 46개의 상호 배타적인 토픽으로 분류하는 신경망
# 클래스가 많기 때문에 이 문제는 다중 분류의 예입니다. 
# 각 데이터 포인트가 정확히 하나의 범주로 분류되기 때문에 좀 더 정확히 말하면 단일 레이블 다중 분류 문제

# 1986년에 로이터에서 공개한 짧은 뉴스 기사와 토픽의 집합인 로이터 데이터셋
# 46개의 토픽이 있으며 어떤 토픽은 다른 것에 비해 데이터가 많습니다. 각 토픽은 훈련 세트에 최소한 10개의 샘플
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(train_data.shape, len(train_data), train_labels.shape, len(train_labels))
print(train_data[0]) # 
print(train_labels) # 

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print("decoded_newswire:", decoded_newswire)

# 
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 훈련 데이터 벡터 변환
x_train = vectorize_sequences(train_data)
# 테스트 데이터 벡터 변환
x_test = vectorize_sequences(test_data)

# 레이블을 벡터로 바꾸는 방법은 두 가지입니다. 레이블의 리스트를 정수 텐서로 변환하는 것과 원-핫 인코딩을 사용하는 것입니다. 
# 원-핫 인코딩이 범주형 데이터에 널리 사용되기 때문에 범주형 인코딩이라고도 부릅니다
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 훈련 레이블 벡터 변환
one_hot_train_labels = to_one_hot(train_labels)
# 테스트 레이블 벡터 변환
one_hot_test_labels = to_one_hot(test_labels)

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers
# 출력 클래스의 개수가 46개
# 16차원 공간은 46개의 클래스를 구분하기에 너무 제약이 많을 것 같습니다. 
# 이렇게 규모가 작은 층은 유용한 정보를 완전히 잃게 되는 정보의 병목 지점처럼 동작할 수 있습니다.
# 이런 이유로 좀 더 규모가 큰 층을 사용하겠습니다. 64개의 유닛을 사용
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 손실 함수 categorical_crossentropy : 두 확률 분포의 사이의 거리를 측정
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. 훈련 검증
# 원본 훈련 데이터에서 10,000의 샘플을 떼어서 검증 세트로
# Train on 15000 samples, validate on 10000 samples 
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 모델을 512개 샘플씩 미니 배치를 만들어 30번의 에포크 동안 훈련
epochsCnt = 30
train_result = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochsCnt,
                    batch_size=512,
                    validation_data=(x_val, y_val))

loss = train_result.history['loss']
val_loss = train_result.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
image.save_fig("3.5.1 Training and validation loss")  
plt.show()

plt.clf()   # 그래프를 초기화합니다
acc = train_result.history['accuracy']
val_acc = train_result.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
image.save_fig("3.5.2 Training and validation accuracy")  

plt.show()

# 이 모델은 9번째 에포크 이후에 과대적합이 시작됨
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print("results", results)

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels)

# 새로운 데이터에 대해 예측하기
predictions = model.predict(x_test)
print("predictions", predictions)
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))

# 레이블과 손실을 다루는 다른 방법 : 정수 텐서로 변환
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 충분히 큰 중간층을 두어야 하는 이유
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_result_dense4 = model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))

results_dense4 = model.evaluate(x_test, one_hot_test_labels)
print("results_dense4", results_dense4)



loss = train_result_dense4.history['loss']
val_loss = train_result_dense4.history['val_loss']
epochs = range(1, len(loss) + 1)
acc = train_result_dense4.history['accuracy']
val_acc = train_result_dense4.history['val_accuracy']

plt.clf()   # 그래프를 초기화합니다
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, acc, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation loss & accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss & Accuracy')
plt.legend()
image.save_fig("3.5.3 Training and validation Loss & Accuracy")  
plt.show()



# 정리
'''
N개의 클래스로 데이터 포인트를 분류하려면 네트워크의 마지막 Dense 층의 크기는 N이어야 합니다.
단일 레이블, 다중 분류 문제에서는 N개의 클래스에 대한 확률 분포를 출력하기 위해 softmax 활성화 함수를 사용해야 합니다.
이런 문제에는 항상 범주형 크로스엔트로피를 사용해야 합니다. 이 함수는 모델이 출력한 확률 분포와 타깃 분포 사이의 거리를 최소화합니다.
다중 분류에서 레이블을 다루는 두 가지 방법이 있습니다.
레이블을 범주형 인코딩(또는 원-핫 인코딩)으로 인코딩하고 categorical_crossentropy 손실 함수를 사용합니다.
레이블을 정수로 인코딩하고 sparse_categorical_crossentropy 손실 함수를 사용합니다.
많은 수의 범주를 분류할 때 중간층의 크기가 너무 작아 네트워크에 정보의 병목이 생기지 않도록 해야 합니다.
'''