import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

import functools
import numpy as np
import tensorflow as tf

# 심장 데이터 셋이 포함 된 csv 파일을 다운로드합니다.

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

df = pd.read_csv(csv_file)
print(df.head())
df.info()

# 변환 thal인 열 object 이산 수치로하여 dataframe.
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
print(df.head())

# tf.data.Dataset 사용하여 데이터로드 
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

# 데이터 세트를 섞고 일괄 처리합니다.
train_dataset = dataset.shuffle(len(df)).batch(1)

# 모델 생성 및 훈련
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)


test_loss, test_accuracy = model.evaluate(train_dataset)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(train_dataset)

# Show some results
for prediction, target in zip(predictions[:10], list(train_dataset)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted target: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("NEGATIVE" if bool(target) else "POSITIVE"))


# 특성 열의 대안
# 사전을 모델에 대한 입력으로 전달하는 것은 tf.keras.layers.Input 레이어의 일치하는 사전을 tf.keras.layers.Input 사전 처리를 적용하고 
# 기능적 api를 사용하여 쌓아 tf.keras.layers.Input 큼 쉽습니다. 
# 이 기능 을 특성 열의 대안으로 사용할 수 있습니다
'''
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
  print (dict_slice)

model_func.fit(dict_slices, epochs=15)
'''





