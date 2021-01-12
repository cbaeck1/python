# 8. 
# 8.1 정형 데이터 다루기
# 목차
# 1. 데이터셋
# 1.1 판다스로 데이터프레임 만들기
# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
# 1.3 tf.data를 사용하여 입력 파이프라인 만들기
# 1.4 입력 파이프라인 이해하기
# 1.5 여러 종류의 특성 열 알아 보기
# 2. 사용할 열 선택하기
# 3. 
# 4. 

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
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# 1. 데이터셋
# 클리블랜드(Cleveland) 심장병 재단에서 제공한 작은 데이터셋
# 이 CSV 파일은 수백 개의 행으로 이루어져 있습니다. 
# 각 행은 환자 한 명을 나타내고 각 열은 환자에 대한 속성 값입니다. 
# 이 정보를 사용해 환자의 심장병 발병 여부를 예측

# 1.1 판다스로 데이터프레임 만들기
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
print(dataframe.head(), dataframe.info(), dataframe.shape)

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print("train : {} {} {} {}".format(len(train), train.shape, type(train), train.info()))
print("test : {} {} {} {}".format(len(test), test.shape, type(test), test.info()))
print("val : {} {} {} {}".format(len(val), val.shape, type(val), val.info()))

# 1.3 tf.data를 사용하여 입력 파이프라인 만들기
# 판다스 데이터프레임으로부터 tf.data 데이터셋을 만들기 위한 함수
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 5 # 예제를 위해 작은 배치 크기를 사용합니다.
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 1.4 입력 파이프라인 이해하기
for feature_batch, label_batch in train_ds.take(1):
  print('전체 특성:', list(feature_batch.keys()))
  print('나이 특성의 배치:', feature_batch['age'])
  print('타깃의 배치:', label_batch )

# 1.5 여러 종류의 특성 열 알아 보기
# 특성 열을 시험해 보기 위해 샘플 배치를 만듭니다.
example_batch = next(iter(train_ds))[0]

# 특성 열을 만들고 배치 데이터를 변환하는 함수
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  try:
    print(feature_column.name, feature_layer(example_batch).numpy())
  except OverflowError:
    print(feature_column.name, 'except:', OverflowError)

# 수치형 열 : 특성 열의 출력은 모델의 입력이 됩니다
# (앞서 정의한 함수를 사용하여 데이터프레임의 각 열이 어떻게 변환되는지 알아 볼 것입니다). 
# 수치형 열은 가장 간단한 종류의 열입니다. 이 열은 실수 특성을 표현하는데 사용됩니다. 
# 이 열을 사용하면 모델은 데이터프레임 열의 값을 변형시키지 않고 그대로 전달 받습니다.
age = feature_column.numeric_column("age")
demo(age)

# 버킷형 열 : 종종 모델에 수치 값을 바로 주입하기 원치 않을 때가 있습니다. 
# 대신 수치 값의 구간을 나누어 이를 기반으로 범주형으로 변환합니다. 
# 원본 데이터가 사람의 나이를 표현한다고 가정해 보죠. 
# 나이를 수치형 열로 표현하는 대신 버킷형 열(bucketized column)을 사용하여 나이를 몇 개의 버킷(bucket)으로 분할할 수 있습니다. 
# 다음에 원-핫 인코딩(one-hot encoding)된 값은 각 열이 매칭되는 나이 범위를 나타냅니다.
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

# 범주형 열 : 이 데이터셋에서 thal 열은 문자열입니다(예를 들어 'fixed', 'normal', 'reversible'). 
# 모델에 문자열을 바로 주입할 수 없습니다. 대신 문자열을 먼저 수치형으로 매핑해야 합니다. 
# 범주형 열(categorical column)을 사용하여 문자열을 원-핫 벡터로 표현할 수 있습니다. 
# 문자열 목록은 categorical_column_with_vocabulary_list를 사용하여 리스트로 전달하거나 
# categorical_column_with_vocabulary_file을 사용하여 파일에서 읽을 수 있습니다.
thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# 임베딩 열 : 가능한 문자열이 몇 개가 있는 것이 아니라 범주마다 수천 개 이상의 값이 있는 경우를 상상해 보겠습니다. 
# 여러 가지 이유로 범주의 개수가 늘어남에 따라 원-핫 인코딩을 사용하여 신경망을 훈련시키는 것이 불가능해집니다. 
# 임베딩 열(embedding column)을 사용하면 이런 제한을 극복할 수 있습니다. 
# 고차원 원-핫 벡터로 데이터를 표현하는 대신 임베딩 열을 사용하여 저차원으로 데이터를 표현합니다. 
# 이 벡터는 0 또는 1이 아니라 각 원소에 어떤 숫자도 넣을 수 있는 밀집 벡터(dense vector)입니다. 
# 임베딩의 크기(아래 예제에서는 8입니다)는 튜닝 대상 파라미터입니다.
# 핵심 포인트: 범주형 열에 가능한 값이 많을 때는 임베딩 열을 사용하는 것이 최선입니다. 
# 여기에서는 예시를 목적으로 하나를 사용하지만 완전한 예제이므로 나중에 다른 데이터셋에 수정하여 적용할 수 있습니다.
# 임베딩 열의 입력은 앞서 만든 범주형 열입니다.
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

# 해시 특성 열 : 가능한 값이 많은 범주형 열을 표현하는 또 다른 방법은 categorical_column_with_hash_bucket을 사용하는 것입니다. 
# 이 특성 열은 입력의 해시(hash) 값을 계산한 다음 hash_bucket_size 크기의 버킷 중 하나를 선택하여 문자열을 인코딩합니다. 
# 이 열을 사용할 때는 어휘 목록을 제공할 필요가 없고 공간을 절약하기 위해 
# 실제 범주의 개수보다 훨씬 작게 해시 버킷(bucket)의 크기를 정할 수 있습니다.
# 핵심 포인트: 이 기법의 큰 단점은 다른 문자열이 같은 버킷에 매핑될 수 있다는 것입니다. 
# 그럼에도 실전에서는 일부 데이터셋에서 잘 작동합니다.
thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

# 교차 특성 열 : 여러 특성을 연결하여 하나의 특성으로 만드는 것을 교차 특성(feature cross)이라고 합니다. 
# 모델이 특성의 조합에 대한 가중치를 학습할 수 있습니다. 이 예제에서는 age와 thal의 교차 특성을 만들어 보겠습니다. 
# crossed_column은 모든 가능한 조합에 대한 해시 테이블을 만들지 않고 
# hashed_column 매개변수를 사용하여 해시 테이블의 크기를 선택합니다.
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

# 2. 사용할 열 선택하기
feature_columns = []
# 수치형 열
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))
# 버킷형 열
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)
# 범주형 열
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)
# 임베딩 열
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)
# 교차 특성 열
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# 2.1 특성 층 만들기 : 특성 열을 정의하고 나면 DenseFeatures 층을 사용해 케라스 모델에 주입할 수 있습니다.
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# 앞서 특성 열의 작동 예를 보이기 위해 작은 배치 크기를 사용했습니다. 여기에서는 조금 더 큰 배치 크기로 입력 파이프라인을 만듭니다.
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# print(train_ds.shape, test_ds.shape, val_ds.shape)
drop_rate = 0.3

# 3.  모델 생성, 컴파일, 훈련
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(512, kernel_initializer='glorot_normal', activation='relu'),
  layers.Dropout(drop_rate),
  layers.Dense(512, kernel_initializer='glorot_normal', activation='relu'), 
  layers.Dropout(drop_rate),
  layers.Dense(512, kernel_initializer='glorot_normal', activation='relu'), 
  layers.Dropout(drop_rate),
  layers.Dense(512, kernel_initializer='glorot_normal', activation='relu'), 
  layers.Dropout(drop_rate),
  layers.Dense(1, kernel_initializer='glorot_normal',  activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# learning_rate = 0.01
# model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
# model.summary()

epoch_size = 500
history = model.fit(train_ds, validation_data=val_ds, epochs=epoch_size, verbose=0)

# print(history.history['loss'])
plt.grid(True)
plt.plot(list(np.arange(epoch_size)), history.history['loss'], label='loss')
plt.plot(list(np.arange(epoch_size)), history.history['accuracy'], label='accuracy')
plt.legend(loc='best')   # center right
images.image.save_fig("tf2.8.1.heart_nn_wrA_loss_accuracy_by_epochs_size")    
plt.show()

loss, accuracy = model.evaluate(test_ds)
print("loss, 정확도", loss, accuracy)
#loss, 정확도 0.47257041931152344 0.7377049326896667

# 핵심 포인트: 일반적으로 크고 복잡한 데이터셋일 경우 딥러닝 모델에서 최선의 결과를 얻습니다. 
# 이런 작은 데이터셋에서는 기본 모델로 결정 트리(decision tree)나 랜덤 포레스트(random forest)를 사용하는 것이 권장됩니다. 
# 이 튜토리얼의 목적은 정확한 모델을 훈련하는 것이 아니라 정형 데이터를 다루는 방식을 설명하는 것입니다. 
# 실전 데이터셋을 다룰 때 이 코드를 시작점으로 사용하세요.

# 4. 그 다음엔
# 정형 데이터를 사용한 분류 작업에 대해 배우는 가장 좋은 방법은 직접 실습하는 것입니다. 
# 실험해 볼 다른 데이터셋을 찾아서 위와 비슷한 코드를 사용해 모델을 훈련해 보세요. 
# 정확도를 향상시키려면 모델에 포함할 특성과 표현 방법을 신중하게 생각하세요.




