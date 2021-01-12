# 자동차 연비 예측하기: 회귀
# 가격이나 확률 같이 연속된 출력 값을 예측하는 것이 목적
# Auto MPG 데이터셋을 사용하여 1970년대 후반과 1980년대 초반의 자동차 연비를 예측하는 모델
# 이 기간에 출시된 자동차 정보를 모델에 제공하겠습니다. 이 정보에는 실린더 수, 배기량, 마력(horsepower), 공차 중량 같은 속성이 포함
# 목차
# 1. Auto MPG 데이터셋
#   1.1. 데이터 구하기
#   1.2. 데이터 정제하기
#   1.3. 데이터셋을 훈련세트와 테스트 세트로 분할하기
#   1.4. 데이터 조사하기
#   1.5. 특성과 레이블 분리하기
#   1.6. 데이터 정규화
# 2. 모델
#   2.1 모델 만들기
#   2.2 모델 확인
#   2.3 모델 훈련
# 3. 예측
# 4. 결론

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
import seaborn as sns
from tensorflow.keras import layers

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#   1.1. 데이터 구하기
dataset_path = keras.utils.get_file("auto-mpg.data", 
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()

# 1.2. 데이터 정제하기 
# 문제를 간단하게 만들기 위해서 누락된 행을 삭제
dataset = dataset.dropna()
# "Origin" 열은 수치형이 아니고 범주형이므로 원-핫 인코딩(one-hot encoding)으로 변환
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

# 1.3. 데이터셋을 훈련세트와 테스트 세트로 분할하기
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 1.4. 데이터 조사하기: 훈련 세트에서 몇 개의 열을 선택해 산점도 행렬
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
images.image.save_fig("1.4Auto MPG dataset")     
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# 1.5. 특성과 레이블 분리하기 : 이 레이블을 예측하기 위해 모델을 훈련시킬 것입니다.
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 1.6. 데이터 정규화
# 위 train_stats 통계를 다시 살펴보고 각 특성의 범위가 얼마나 다른지 확인해 보죠.
# 특성의 스케일과 범위가 다르면 정규화(normalization)하는 것이 권장됩니다. 
# 특성을 정규화하지 않아도 모델이 수렴할 수 있지만, 훈련시키기 어렵고 입력 단위에 의존적인 모델이 만들어집니다.
# 노트: 의도적으로 훈련 세트만 사용하여 통계치를 생성했습니다. 
# 이 통계는 테스트 세트를 정규화할 때에도 사용됩니다. 이는 테스트 세트를 모델이 훈련에 사용했던 것과 동일한 분포로 투영하기 위해서입니다.
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 정규화된 데이터를 사용하여 모델을 훈련합니다.
# 주의: 여기에서 입력 데이터를 정규화하기 위해 사용한 통계치(평균과 표준편차)는 원-핫 인코딩과 마찬가지로 
# 모델에 주입되는 모든 데이터에 적용되어야 합니다. 
# 여기에는 테스트 세트는 물론 모델이 실전에 투입되어 얻은 라이브 데이터도 포함됩니다.

# 2. 모델
# 2.1 모델 만들기
#    두 개의 완전 연결(densely connected) 은닉층으로 Sequential 모델
#    출력 층은 하나의 연속적인 값을 반환합니다. 
#    나중에 두 번째 모델을 만들기 쉽도록 build_model 함수로 모델 구성 단계를 감싸겠습니다.
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# 2.2 모델 확인
# .summary 메서드를 사용해 모델에 대한 간단한 정보를 출력합니다.
model.summary()

# 모델을 한번 실행해 보죠. 훈련 세트에서 10 샘플을 하나의 배치로 만들어 model.predict 메서드를 호출해 보겠습니다.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

# 2.3 모델 훈련
# 이 모델을 1,000번의 에포크(epoch) 동안 훈련합니다. 훈련 정확도와 검증 정확도는 history 객체에 기록됩니다.
# 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시합니다
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# history 객체에 저장된 통계치를 사용해 모델의 훈련 과정을 시각화해 보죠.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  #images.image.save_fig(hist)     
  plt.show()

plot_history(history)

model = build_model()

# patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))

# 3. 예측
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
images.image.save_fig('1.4test_predictions')     
plt.show()

# 모델이 꽤 잘 예측한 것 같습니다. 오차의 분포를 살펴 보죠.
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
images.image.save_fig('1.4error_distribution')    
plt.show()

# 가우시안 분포가 아니지만 아마도 훈련 샘플의 수가 매우 작기 때문일 것입니다.

# 4. 결론
# 이 노트북은 회귀 문제를 위한 기법을 소개합니다.
# 평균 제곱 오차(MSE)는 회귀 문제에서 자주 사용하는 손실 함수입니다(분류 문제에서 사용하는 손실 함수와 다릅니다).
# 비슷하게 회귀에서 사용되는 평가 지표도 분류와 다릅니다. 많이 사용하는 회귀 지표는 평균 절댓값 오차(MAE)입니다.
# 수치 입력 데이터의 특성이 여러 가지 범위를 가질 때 동일한 범위가 되도록 각 특성의 스케일을 독립적으로 조정해야 합니다.
# 훈련 데이터가 많지 않다면 과대적합을 피하기 위해 은닉층의 개수가 적은 소규모 네트워크를 선택하는 방법이 좋습니다.
# 조기 종료(Early stopping)은 과대적합을 방지하기 위한 좋은 방법입니다.



