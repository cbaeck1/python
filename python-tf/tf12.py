# 영화 리뷰를 사용한 텍스트 분류 : 
# 영화 리뷰(review) 텍스트를 긍정(positive) 또는 부정(negative)으로 분류
# 목차
# 1. IMDB 데이터셋 
# 2. 데이터 탐색 : 단어를 정수로, 정수를 단어로
# 3. 데이터 준비
# 4. 모델구성 
#   4.1 은닉 유닛
#   4.2 손실함수와 옴티마이저
# 5. 검증세트 만들기
# 6. 모델훈련
# 7. 모델평가
# 8. 정확도와 손실 그래프그리기

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image
print(tf.__version__)

# 1. IMDB 데이터셋 다운로드
# 리뷰(단어의 시퀀스(sequence))는 미리 전처리해서 정수 시퀀스로 변환되어 있습니다. 각 정수는 어휘 사전에 있는 특정 단어를 의미
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 매개변수 num_words=10000은 훈련 데이터에서 가장 많이 등장하는 상위 10,000개의 단어를 선택합니다.
# 데이터 크기를 적당하게 유지하기 위해 드물 등장하는 단어는 제외

# 2. 데이터 탐색 
# 이 데이터셋의 샘플은 전처리된 정수 배열입니다. 이 정수는 영화 리뷰에 나오는 단어를 나타냅니다. 
# 레이블(label)은 정수 0 또는 1입니다. 0은 부정적인 리뷰이고 1은 긍정적인 리뷰
print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))
print("train_data[0]", train_data[0])

# 영화 리뷰들은 길이가 다릅니다. 다음 코드는 첫 번째 리뷰와 두 번째 리뷰에서 단어의 개수를 출력합니다. 
# 신경망의 입력은 길이가 같아야 하기 때문에 나중에 이 문제를 해결해야 함
print(len(train_data[0]), len(train_data[1]))

# 정수를 단어로 다시 변환하기
# 정수와 문자열을 매핑한 딕셔너리(dictionary) 객체에 질의하는 헬퍼(helper) 함수를 만들겠습니다:
# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()

# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(textNumber):
    return ' '.join([reverse_word_index.get(i, '?') for i in textNumber])

print(decode_review(train_data[0]))
print(decode_review(train_data[1]))

# 3. 데이터 준비
# 리뷰-정수 배열-는 신경망에 주입하기 전에 텐서로 변환
# 1) 원-핫 인코딩(one-hot encoding)은 정수 배열을 0과 1로 이루어진 벡터로 변환
# 예를 들어 배열 [3, 5]을 인덱스 3과 5만 1이고 나머지는 모두 0인 10,000차원 벡터로 변환
# 그다음 실수 벡터 데이터를 다룰 수 있는 층-Dense 층-을 신경망의 첫 번째 층으로 사용
# 이 방법은 num_words * num_reviews 크기의 행렬이 필요하기 때문에 메모리를 많이 사용
# 2) 정수 배열의 길이가 모두 같도록 패딩(padding)을 추가해 max_length * num_reviews 크기의 정수 텐서로 
# 이런 형태의 텐서를 다룰 수 있는 임베딩(embedding) 층을 신경망의 첫 번째 층으로 사용

# 이 튜토리얼에서는 두 번째 방식을 사용하겠습니다.
# 영화 리뷰의 길이가 같아야 하므로 pad_sequences 함수를 사용해 길이를 맞추겠습니다:
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
    padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
    padding='post', maxlen=256)

# 샘플의 길이를 확인
print(len(train_data[0]), len(train_data[1]))

# (패딩된) 첫 번째 리뷰 내용을 확인해 보겠습니다:
print(train_data[0])
print(train_data[1])

# 4. 모델 구성
# 신경망은 층(layer)을 쌓아서 만듭니다. 이 구조에서는 두 가지를 결정해야 합니다:
# 1) 모델에서 얼마나 많은 층을 사용할 것인가?
# 2) 각 층에서 얼마나 많은 은닉 유닛(hidden unit)을 사용할 것인가?
# 이 예제의 입력 데이터는 단어 인덱스의 배열입니다. 예측할 레이블(출력층)은 0 또는 1입니다.
# 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기입니다(10,000개의 단어)
# 워드 임베딩이란 텍스트 내의 단어들을 밀집 벡터(dense vector)로 (값의 타입이 실수)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,))) # 입력층
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))  # 은닉층
model.add(keras.layers.Dense(1, activation='sigmoid')) # 출력층

print("model", model.summary())

# 층을 순서대로 쌓아 분류기(classifier)를 만듭니다:
# 첫 번째 층은 Embedding 층입니다. 이 층은 정수로 인코딩된 단어를 입력 받고 각 단어 인덱스에 해당하는 임베딩 벡터를 찾습니다.
# 이 벡터는 모델이 훈련되면서 학습. 출력 배열에 새로운 차원으로 추가
# 최종 차원은 (batch, sequence, embedding)이 됩니다.
# 그다음 GlobalAveragePooling1D 층은 sequence 차원에 대해 평균을 계산, 각 샘플에 대해 고정된 길이의 출력 벡터를 반환
# 이는 길이가 다른 입력을 다루는 가장 간단한 방법입니다.
# 이 고정 길이의 출력 벡터는 16개의 은닉 유닛을 가진 완전 연결(fully-connected) 층(Dense)을 거칩니다.
# 마지막 층은 하나의 출력 노드(node)를 가진 완전 연결 층입니다. 
# sigmoid activation 함수를 사용하여 0과 1 사이의 실수를 출력합니다. 이 값은 확률 또는 신뢰도를 나타냅니다.

# 은닉 유닛 : 위 모델에는 입력과 출력 사이에 두 개의 중간 또는 "은닉" 층이 있습니다. 
# 출력(유닛 또는 노드, 뉴런)의 개수는 층이 가진 표현 공간(representational space)의 차원이 됩니다. 
# 다른 말로 하면, 내부 표현을 학습할 때 허용되는 네트워크 자유도의 양입니다.
# 모델에 많은 은닉 유닛(고차원의 표현 공간)과 층이 있다면 네트워크는 더 복잡한 표현을 학습할 수 있습니다. 
# 하지만 네트워크의 계산 비용이 많이 들고 원치않는 패턴을 학습할 수도 있습니다. 
# 이런 표현은 훈련 데이터의 성능을 향상시키지만 테스트 데이터에서는 그렇지 못합니다. 
# 이를 과대적합(overfitting)이라고 부릅니다. 나중에 이에 대해 알아 보겠습니다.
# 손실 함수와 옵티마이저 : 모델이 훈련하려면 손실 함수(loss function)과 옵티마이저(optimizer)가 필요합니다. 
# 이진 분류 문제이고 모델이 확률을 출력하므로(출력층의 유닛이 하나, sigmoid activation 함수를 사용) 
# binary_crossentropy 손실 함수를 사용
# 다른 손실 함수(예 mean_squared_error)를 선택할 수 있습니다. 
# 하지만 일반적으로 binary_crossentropy가 확률을 다루는데 적합합니다. 이 함수는 확률 분포 간의 거리를 측정합니다. 
# 여기에서는 정답인 타깃 분포와 예측 분포 사이의 거리입니다.
# 나중에 회귀(regression) 문제(예 주택 가격을 예측하는 문제)에 대해 평균제곱오차(mean squared error) 손실 함수사용

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. 검증 세트 만들기
# 모델을 훈련할 때 모델이 만난 적 없는 데이터에서 정확도를 확인하는 것이 좋습니다. 
# 원본 훈련 데이터에서 10,000개의 샘플을 떼어내어 검증 세트(validation set)를 만들겠습니다. 
# (왜 테스트 세트를 사용하지 않을까요? 훈련 데이터만을 사용하여 모델을 개발하고 튜닝하는 것이 목표입니다. 
# 그다음 테스트 세트를 사용해서 딱 한 번만 정확도를 평가합니다).
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 6. 모델 훈련
# 이 모델을 512개의 샘플로 이루어진 미니배치(mini-batch)에서 40번의 에포크(epoch) 동안 훈련합니다. 
# x_train과 y_train 텐서에 있는 모든 샘플에 대해 40번 반복한다는 뜻입니다.
# 훈련하는 동안 10,000개의 검증 세트에서 모델의 손실과 정확도를 모니터링합니다:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 6. 모델 평가
# 모델의 성능을 확인해 보죠. 두 개의 값이 반환됩니다. 손실(오차를 나타내는 숫자이므로 낮을수록 좋습니다)과 정확도입니다.
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

# 이 예제는 매우 단순한 방식을 사용하므로 87% 정도의 정확도를 달성했습니다. 고급 방법을 사용한 모델은 95%에 가까운 정확도를 얻습니다.
# 정확도와 손실 그래프 그리기
# model.fit()은 History 객체를 반환합니다. 여기에는 훈련하는 동안 일어난 모든 정보가 담긴 딕셔너리(dictionary)가 들어 있습니다:
history_dict = history.history
history_dict.keys()

# 네 개의 항목이 있습니다. 훈련과 검증 단계에서 모니터링하는 지표들입니다. 
# 훈련 손실과 검증 손실을 그래프로 그려 보고, 훈련 정확도와 검증 정확도도 그래프로 그려서 비교해 보겠습니다:
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
images.image.save_fig("1.2imdb_train_validation_loss")     
plt.show()


plt.clf()   # 그림을 초기화합니다
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
images.image.save_fig("1.2imdb_train_validation_accuracy")     
plt.show()

# 이 그래프에서 점선은 훈련 손실과 훈련 정확도를 나타냅니다. 실선은 검증 손실과 검증 정확도입니다.
# 훈련 손실은 에포크마다 감소하고 훈련 정확도는 증가한다는 것을 주목하세요. 
# 경사 하강법 최적화를 사용할 때 볼 수 있는 현상입니다. 매 반복마다 최적화 대상의 값을 최소화합니다.
# 하지만 검증 손실과 검증 정확도에서는 그렇지 못합니다. 약 20번째 에포크 이후가 최적점인 것 같습니다. 이는 과대적합 때문입니다. 
# 이전에 본 적 없는 데이터보다 훈련 데이터에서 더 잘 동작합니다. 
# 이 지점부터는 모델이 과도하게 최적화되어 테스트 데이터에서 일반화되기 어려운 훈련 데이터의 특정 표현을 학습합니다.
# 여기에서는 과대적합을 막기 위해 단순히 20번째 에포크 근처에서 훈련을 멈출 수 있습니다.
# 나중에 콜백(callback)을 사용하여 자동으로 이렇게 하는 방법을 배워 보겠습니다.

