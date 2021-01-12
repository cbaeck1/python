# 케라스와 텐서플로 허브를 사용한 영화 리뷰 텍스트 분류하기 
# 영화 리뷰(review) 텍스트를 긍정(positive) 또는 부정(negative)으로 분류
# 텐서플로 허브(TensorFlow Hub)와 케라스(Keras)를 사용한 기초적인 전이 학습(transfer learning) 애플리케이션
# 목차
# 1. IMDB 데이터셋 
# 2. 데이터 탐색
# 3. 모델구성 
#   3.1 손실함수와 옴티마이저
# 4. 모델훈련
# 5. 모델평가

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras
import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# 1. IMDB 데이터셋 다운로드
# 훈련 세트를 6대 4로 : 훈련에 15,000개 샘플, 검증에 10,000개 샘플, 테스트에 25,000개 샘플을 사용하게 됩니다.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# 2. 데이터 탐색 : 
# 이 데이터셋의 샘플은 전처리된 정수 배열입니다. 이 정수는 영화 리뷰에 나오는 단어를 나타냅니다. 
# 레이블(label)은 정수 0 또는 1입니다. 0은 부정적인 리뷰이고 1은 긍정적인 리뷰
# 처음 10개의 샘플을 출력해 보겠습니다.
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)


# 3. 모델 구성
# 신경망은 층을 쌓아서 만듭니다. 여기에는 세 개의 중요한 구조적 결정이 필요합니다:
# 어떻게 텍스트를 표현할 것인가?
# 모델에서 얼마나 많은 층을 사용할 것인가?
# 각 층에서 얼마나 많은 은닉 유닛(hidden unit)을 사용할 것인가?
# 이 예제의 입력 데이터는 문장으로 구성됩니다. 예측할 레이블은 0 또는 1입니다.

# 텍스트를 표현하는 방법으로 문장을 임베딩(embedding) 벡터로 바꾸면 
# 첫 번째 층으로 사전 훈련(pre-trained)된 텍스트 임베딩을 사용할 수 있습니다.
# 텍스트 전처리에 대해 신경 쓸 필요가 없습니다.
# 전이 학습의 장점을 이용합니다.
# 임베딩은 고정 크기이기 때문에 처리 과정이 단순해집니다.
# 이 예제는 텐서플로 허브에 있는 사전 훈련된 텍스트 임베딩 모델인 google/tf2-preview/gnews-swivel-20dim/1을 사용하겠습니다.
# 테스트해 볼 수 있는 사전 훈련된 모델이 세 개 더 있습니다:
#  1) google/tf2-preview/gnews-swivel-20dim-with-oov/1 
#     google/tf2-preview/gnews-swivel-20dim/1와 동일하지만 어휘 사전(vocabulary)의 2.5%가 OOV 버킷(bucket)으로 변환되었습니다. 
#     이는 해당 문제의 어휘 사전과 모델의 어휘 사전이 완전히 겹치지 않을 때 도움이 됩니다.
#  2) google/tf2-preview/nnlm-en-dim50/1 : 더 큰 모델. 차원 크기는 50이고 어휘 사전의 크기는 1백만 개 이하입니다.
#  3) google/tf2-preview/nnlm-en-dim128/1 : 훨씬 더 큰 모델. 차원 크기는 128이고 어휘 사전의 크기는 1백만 개 이하입니다.
# 먼저 문장을 임베딩시키기 위해 텐서플로 허브 모델을 사용하는 케라스 층을 만들어 보죠.
# 그다음 몇 개의 샘플을 입력하여 테스트해 보겠습니다. 
# 입력 텍스트의 길이에 상관없이 임베딩의 출력 크기는 (num_examples, embedding_dimension)가 됩니다.
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
print(model.summary())


# 순서대로 층을 쌓아 분류기를 만듭니다:
# 첫 번째 층은 텐서플로 허브 층입니다. 이 층은 사전 훈련된 모델을 사용하여 하나의 문장을 임베딩 벡터로 매핑합니다. 여기서 사용하는 사전 훈련된 텍스트 임베딩 모델(google/tf2-preview/gnews-swivel-20dim/1)은 하나의 문장을 토큰(token)으로 나누고 각 토큰의 임베딩을 연결하여 반환합니다. 최종 차원은 (num_examples, embedding_dimension)입니다.
# 이 고정 크기의 출력 벡터는 16개의 은닉 유닛(hidden unit)을 가진 완전 연결 층(Dense)으로 주입됩니다.
# 마지막 층은 하나의 출력 노드를 가진 완전 연결 층입니다. sigmoid 활성화 함수를 사용하므로 확률 또는 신뢰도 수준을 표현하는 0~1 사이의 실수가 출력됩니다.
# 이제 모델을 컴파일합니다.
# 3.1 손실 함수와 옵티마이저
# 모델이 훈련하려면 손실 함수(loss function)과 옵티마이저(optimizer)가 필요합니다. 
# 이 예제는 이진 분류 문제이고 모델이 확률을 출력하므로(출력층의 유닛이 하나이고 sigmoid 활성화 함수를 사용합니다),
# binary_crossentropy 손실 함수를 사용하겠습니다.
# 다른 손실 함수를 선택할 수 없는 것은 아닙니다. 예를 들어 mean_squared_error를 선택할 수 있습니다. 
# 하지만 일반적으로 binary_crossentropy가 확률을 다루는데 적합합니다. 이 함수는 확률 분포 간의 거리를 측정합니다. 
# 여기에서는 정답인 타깃 분포와 예측 분포 사이의 거리입니다.
# 나중에 회귀(regression) 문제(예를 들어 주택 가격을 예측하는 문제)에 대해 살펴 볼 때 
# 평균 제곱 오차(mean squared error) 손실 함수를 어떻게 사용하는지 알아 보겠습니다.

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. 모델 훈련
# 이 모델을 512개의 샘플로 이루어진 미니배치(mini-batch)에서 20번의 에포크(epoch) 동안 훈련합니다. 
# x_train과 y_train 텐서에 있는 모든 샘플에 대해 20번 반복한다는 뜻입니다. 
# 훈련하는 동안 10,000개의 검증 세트에서 모델의 손실과 정확도를 모니터링합니다:
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 5. 모델 평가
# 모델의 성능을 확인해 보죠. 두 개의 값이 반환됩니다. 손실(오차를 나타내는 숫자이므로 낮을수록 좋습니다)과 정확도입니다.
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

# 이 예제는 매우 단순한 방식을 사용하므로 87% 정도의 정확도를 달성했습니다. 고급 방법을 사용한 모델은 95%에 가까운 정확도를 얻습니다.
