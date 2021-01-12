# 4.1 텐서와 연산
# 목차
# 1. 텐서
# 2. GPU 가속
# 3. 데이터셋

import tensorflow as tf

# 1. 텐서
# 텐서는 다차원 배열입니다. 넘파이(NumPy) ndarray 객체와 비슷
# 1차원 텐서는 벡터이고 코드 상에서 [1, 2, 3]과 같은 1차원 배열입니다. 
# 2차원 텐서는 행렬이며 코드상에서 [[1, 2 ,3], [4, 5, 6]]과 같은 2차원 배열입니다. 
# 3차원 텐서는 행렬을 여러 층 쌓은 것이고 코드상에서 [[[1, 2] ,[3, 4]], [[4, 5], [6, 7]]]와 같은 3차원 배열입니다.
# tf.Tensor 객체는 데이터 타입과 크기
# 또한 tf.Tensor는 GPU 같은 가속기 메모리에 상주
# 텐서플로는 텐서를 생성하고 이용하는 풍부한 연산 라이브러리(tf.add, tf.matmul, tf.linalg.inv 등.)를 제공합니다. 
# 이러한 연산은 자동으로 텐서를 파이썬 네이티브(native) 타입으로 변환합니다.
# 예를 들어:
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# 연산자 오버로딩(overloading) 또한 지원합니다.
print(tf.square(2) + tf.square(3))

# 각각의 tf.Tensor는 크기와 데이터 타입을 가지고 있습니다.
# 3차원 텐서 사이의 행렬 곱
x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)

# 넘파이 배열과 tf.Tensor의 차이
# 텐서는 가속기 메모리(GPU, TPU와 같은)에서 사용할 수 있습니다.
# 텐서는 불변성(immutable)을 가집니다.
# 1.1 넘파이 호환성
# 텐서플로 연산은 자동으로 넘파이 배열을 텐서로 변환합니다.
# 넘파이 연산은 자동으로 텐서를 넘파이 배열로 변환합니다.
# 텐서는 .numpy() 메서드(method)를 호출하여 넘파이 배열로 변환할 수 있습니다. 
# 가능한 경우, tf.Tensor와 배열은 메모리 표현을 공유하기 때문에 이러한 변환은 일반적으로 간단(저렴)합니다. 
# 그러나 tf.Tensor는 GPU 메모리에 저장될 수 있고, 넘파이 배열은 항상 호스트 메모리에 저장되므로, 
# 이러한 변환이 항상 가능한 것은 아닙니다. 따라서 GPU에서 호스트 메모리로 복사가 필요합니다.
import numpy as np
ndarray = np.ones([3, 3])
print("텐서플로 연산은 자동적으로 넘파이 배열을 텐서로 변환합니다.")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print("그리고 넘파이 연산은 자동적으로 텐서를 넘파이 배열로 변환합니다.")
print(np.add(tensor, 1))
print(".numpy() 메서드는 텐서를 넘파이 배열로 변환합니다.")
print(tensor.numpy())

# 2. GPU 가속
# 대부분의 텐서플로 연산은 GPU를 사용하여 가속화됩니다. 
# 어떠한 코드를 명시하지 않아도, 텐서플로는 연산을 위해 CPU 또는 GPU를 사용할 것인지를 자동으로 결정합니다. 
# 필요시 텐서를 CPU와 GPU 메모리 사이에서 복사합니다. 
# 연산에 의해 생성된 텐서는 전형적으로 연산이 실행된 장치의 메모리에 의해 실행됩니다. 
# 예를 들어:
x = tf.random.uniform([3, 3])
print("GPU 사용이 가능한가 : "),
print(tf.test.is_gpu_available())
print("텐서가 GPU #0에 있는가 : "),
print(x.device.endswith('GPU:0'))

# 2.1 장치 이름
# Tensor.device는 텐서를 구성하고 있는 호스트 장치의 풀네임을 제공합니다. 
# 이러한 이름은 프로그램이 실행중인 호스트의 네트워크 주소 및 해당 호스트 내의 장치와 같은 많은 세부 정보를 인코딩하며, 
# 이것은 텐서플로 프로그램의 분산 실행에 필요합니다. 텐서가 호스트의 N번째 GPU에 놓여지면 문자열은 GPU:<N>으로 끝납니다.

# 2.2 명시적 장치 배치
# 텐서플로에서 "배치(replacement)"는 개별 연산을 실행하기 위해 장치에 할당(배치)하는 것입니다. 
# 앞서 언급했듯이, 명시적 지침이 없을 경우 텐서플로는 연산을 실행하기 위한 장치를 자동으로 결정하고, 필요시 텐서를 장치에 복사합니다. 
# 그러나 텐서플로 연산은 tf.device을 사용하여 특정한 장치에 명시적으로 배치할 수 있습니다. 
# 예를 들어:
import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# CPU에서 강제 실행합니다.
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# GPU #0가 이용가능시 GPU #0에서 강제 실행합니다.
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)


# 3. 데이터셋
# 이번에는 모델에 데이터를 제공하기 위한 파이프라인을 구축하기 위해 tf.data.Dataset API를 사용해볼 것입니다.
# tf.data.Dataset API는 모델을 훈련시키고 평가 루프를 제공할, 간단하고 재사용 가능한 모듈로부터 복잡한 입력 파이프라인을 구축하기 위해 사용됩니다.
# 3.1 소스 데이터셋 생성
# 굉장히 유용한 함수중 하나인 Dataset.from_tensors, Dataset.from_tensor_slices와 같은 팩토리(factory) 함수 중 하나를 사용하거나 
# 파일로부터 읽어들이는 객체인 TextLineDataset 또는 TFRecordDataset를 사용하여 소스 데이터셋을 생성하세요. 
# 더 많은 정보를 위해서 텐서플로 데이터셋 가이드를 참조하세요.
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# CSV 파일을 생성합니다.
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

# 3.2 변환 적용
# 맵(map), 배치(batch), 셔플(shuffle)과 같은 변환 함수를 사용하여 데이터셋의 레코드에 적용하세요.
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# 3.3 반복
# tf.data.Dataset은 레코드 순회를 지원하는 반복가능한 객체입니다.
print('ds_tensors 요소:')
for x in ds_tensors:
  print(x)

print('\nds_file 요소:')
for x in ds_file:
  print(x)


