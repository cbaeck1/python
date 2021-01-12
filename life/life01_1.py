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

# 데이터 로드와 사전처리 데이터 : CSV 데이터로드
# 데이터로드
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

print("train_file_path:", train_file_path)

# pandas를 사용하여 이를 로드하고 NumPy 배열을 TensorFlow에 전달
# 명시적으로 식별해야하는 유일한 열은 모델이 예측하려는 값이 있는 열
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

# ( 많은 예 , 많은 라벨 ) 튜플로 표시
def get_dataset(file_path, **kwargs):
  print(file_path)
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)  
  return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):
    print("{:20s}: {}".format(LABEL_COLUMN, label))
    for key, value in batch.items():
      print("{:20s}: {}".format(key, value.numpy()))

raw_train_data = get_dataset(train_file_path)
show_batch(raw_train_data)
raw_test_data = get_dataset(test_file_path)
show_batch(raw_test_data)

# 사용 가능한 모든 열을 사용
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)
show_batch(temp_dataset)

# 사용하려는 열 목록 만 생성
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']
temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
show_batch(temp_dataset)

# 데이터 전처리 : tf.feature_column 
# 원하는 도구 (예 : nltk 또는 sklearn )를 사용하여 데이터를 전처리
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path, 
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)
show_batch(temp_dataset)

# iter() 함수: 반복 가능한 데이터를 입력받아 반복자를 반환하는 함수
# next() 함수: 반복자를 입력받아 다음 출력값을 반환하는 함수
example_batch, labels_batch = next(iter(temp_dataset)) 

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
  print(features.numpy())
  print()
  print(labels.numpy())

show_batch(raw_train_data)
example_batch, labels_batch = next(iter(temp_dataset)) 

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
show_batch(packed_train_data)

example_batch, labels_batch = next(iter(packed_train_data)) 

desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
print("desc:", desc)

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
print("numeric_column:", numeric_column)

example_batch['numeric']
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()

# 범주형 데이터
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

print("categorical_columns:", categorical_columns)  

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])

# 결합 된 전처리 레이어
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=20)
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

