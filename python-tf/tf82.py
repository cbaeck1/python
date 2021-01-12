# 8. 
# 8.2 불균형 데이터 분류
# 한 클래스의 예제 수가 다른 클래스의 예제보다 훨씬 많은 불균형이 심한 데이터 세트를 분류하는 방법을 보여줍니다. 
# Kaggle에서 호스팅되는 신용 카드 사기 감지 데이터 세트를 사용합니다. 
# 목표는 총 284,807 건의 거래에서 492 건의 사기 거래 만 탐지하는 것입니다. 
# Keras 를 사용하여 모델이 불균형 데이터에서 학습하는 데 도움이되는 모델 및 클래스 가중치 를 정의

# 목차
# 1. 설정
# 2. 데이터 처리 및 탐색
# 2.1 Kaggle 신용 카드 사기 데이터 세트 다운로드
# 2.2 클래스 레이블 불균형 조사   
# 2.3 데이터 정리, 분할 및 정규화 
# 2.4 데이터 분포 살펴보기
# 3. 모델 및 메트릭 정의
# 3.1 모델 및 메트릭 정의
# 3.2 유용한 메트릭 이해  
# 4. 기준 모델
# 4.1 모델 구축 
# 5. 

import tensorflow as tf
from tensorflow import keras
import os, sys
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# 2.1 Kaggle 신용 카드 사기 데이터 세트 다운로드
# 이 데이터 세트는 Worldline과 ULB의 Machine Learning Group (Université Libre de Bruxelles)이 
# 빅 데이터 마이닝 및 사기 탐지에 대해 공동 연구하는 동안 수집 및 분석
# cf) https://www.researchgate.net/project/Fraud-detection-with-machine-learning
file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
print(raw_df.shape, raw_df.head())

print(raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe())

# 2.2 클래스 레이블 불균형 조사   
neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# 2.3 데이터 정리, 분할 및 정규화 
# Time 및 Amount 열은 너무 가변적이어서 직접 사용할 수 없습니다. 
# Time 열을 삭제 (의미가 명확하지 않기 때문에) 
# Amount 열의 로그를 가져와 범위를 줄입니다.

cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps=0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

# 데이터 세트를 학습, 검증 및 테스트 세트로 분할합니다. 
# 검증 세트는 모델 피팅 중에 손실 및 메트릭을 평가하는 데 사용되지만 모델이이 데이터에 적합하지 않습니다. 
# 테스트 세트는 훈련 단계에서 완전히 사용되지 않으며 모델이 새 데이터로 얼마나 잘 일반화되는지 평가하기 위해 마지막에만 사용됩니다. 
# 이는 훈련 데이터 부족으로 인해 과적 합 이 중요한 문제인 불균형 데이터 세트에서 특히 중요
# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# sklearn StandardScaler를 사용하여 입력 기능을 정규화합니다. 
# 이것은 평균을 0으로, 표준 편차를 1로 설정합니다.
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

# 2.4 데이터 분포 살펴보기
# 다음으로 몇 가지 기능에 대한 긍정 및 부정 예제의 분포를 비교하십시오. 
# 이 시점에서 스스로에게 물어볼 좋은 질문은 다음과 같습니다.
# 이러한 분포가 의미가 있습니까?
# 예. 입력을 정규화했으며 대부분 +/- 2 범위에 집중되어 있습니다.
# 분포의 차이를 볼 수 있습니까?
# 예, 긍정적 인 예는 훨씬 더 높은 극단 값 비율을 포함합니다.

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

sns.jointplot(pos_df['V5'], pos_df['V6'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
plt.suptitle("Positive distribution")
images.image.save_fig("8.2.Positive_distribution")     
plt.show()

sns.jointplot(neg_df['V5'], neg_df['V6'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
_ = plt.suptitle("Negative distribution")
images.image.save_fig("8.2.Negative_distribution")     
plt.show()

# 3. 모델 및 메트릭 정의 
# 3.1 모델 및 메트릭 정의
# 촘촘하게 연결된 히든 레이어, 과적 합을 줄이기위한 드롭 아웃 레이어, 거래 사기 가능성을 반환하는 출력 시그 모이 드 레이어로 
# 간단한 신경망을 생성하는 함수를 정의합니다.
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model


# 3.2 유용한 메트릭 이해  
# 성능을 평가할 때 도움이 될 모델에서 계산할 수있는 위에 정의 된 몇 가지 메트릭이 있습니다.

# 거짓 음성 및 거짓 양성은 잘못 분류 된 샘플입니다.
# 참 음성 및 참 양성은 올바르게 분류 된 샘플입니다.
# 정확도 는 올바르게 분류 된 예제의 비율입니다.> 
# 정밀도 는 올바르게 분류 된 예측 긍정 비율> 
# 재현율 은 올바르게 분류 된 실제 긍정 비율> 
# AUC 는 ROC-AUC (수신기 작동 특성 곡선)의 곡선 아래 영역을 나타냅니다. 
# 이 측정 항목은 분류 기가 무작위 음성 샘플보다 무작위 양성 샘플의 순위를 매길 확률과 동일합니다.

# 4. 기준 모델
# 4.1 모델 구축 
# 이제 이전에 정의한 함수를 사용하여 모델을 만들고 학습 시키십시오. 
# 모델이 기본 배치 크기 인 2048보다 큰 크기를 사용하여 적합하다는 점에 유의하십시오. 
# 이는 각 배치가 몇 개의 양성 샘플을 포함 할 수있는 적절한 기회를 갖도록하는 데 중요합니다. 
# 배치 크기가 너무 작 으면 배울 수있는 사기 거래가 없을 가능성이 높습니다.
# 참고 : 이 모델은 클래스 불균형을 잘 처리하지 못합니다. 이 자습서의 뒷부분에서 개선합니다.

EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model()
model.summary()

# 모델 테스트 실행 :
model.predict(train_features[:10])

# 선택 사항 : 올바른 초기 바이어스를 설정합니다.
# 이러한 초기 추측은 좋지 않습니다. 데이터 세트가 불균형하다는 것을 알고 있습니다. 이를 반영하도록 출력 계층의 편향을 설정합니다 
# (참조:신경망 훈련을위한 레시피 : "init well" ). 이것은 초기 수렴에 도움이 될 수 있습니다.
# 기본 바이어스 초기화를 사용하면 손실은 대략 math.log(2) = 0.69314
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))



