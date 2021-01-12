# 맞춤설정 : 텐서와 연산 
#   필요한 패키지 임포트
#   텐서(Tensor) 생성 및 사용
#   GPU 가속기 사용
#   tf.data.Dataset 시연
# 
# 목차
# 1. 텐서플로 임포트
# 2. 텐서
#   2.1 Numpy 호환성
# 3. GPU 가속
#   3.1 장치이름
#   3.2 명시적 장치 배제
# 4. 데이터셋
#   4.1 소스 데이터셋 생성
#   4.2 변환 적용
#   4.3 반복


# 1. 텐서플로 임포트 : 텐서플로 2.0에서는 즉시 실행(eager execution)이 기본적으로 실행
import tensorflow as tf

# 2. 텐서
# 텐서는 다차원 배열입니다. 넘파이(NumPy) ndarray 객체와 비슷하며, tf.Tensor 객체는 데이터 타입과 크기를 가지고 있습니다. 
# 또한 tf.Tensor는 GPU 같은 가속기 메모리에 상주할 수 있습니다. 
# 텐서플로는 텐서를 생성하고 이용하는 풍부한 연산 라이브러리(tf.add, tf.matmul, tf.linalg.inv 등.)를 제공합니다. 
# 이러한 연산은 자동으로 텐서를 파이썬 네이티브(native) 타입으로 변환합니다.
print(tf.add(1, 2))                         # tf.Tensor(3, shape=(), dtype=int32)
print(tf.add([1, 2], [3, 4]))               # tf.Tensor([4 6], shape=(2,), dtype=int32)
print(tf.square(5))                         # tf.Tensor(25, shape=(), dtype=int32)
print(tf.reduce_sum([1, 2, 3]))             # tf.Tensor(6, shape=(), dtype=int32)
# 연산자 오버로딩(overloading) 또한 지원합니다.
print(tf.square(2) + tf.square(3))          # tf.Tensor(13, shape=(), dtype=int32)

# 각각의 tf.Tensor는 크기와 데이터 타입을 가지고 있습니다.
# 텐서 사이의 행렬 곱 : matmul
x = tf.matmul([[100]], [[2, 3]]) 
print(x)                                    # tf.Tensor([[200 300]], shape=(1, 2), dtype=int32)
print(x.shape)                              # (1, 2)
print(x.dtype)                              # <dtype: 'int32'>

# 넘파이 배열과 tf.Tensor 의 가장 확연한 차이는 다음과 같습니다
# 텐서는 가속기 메모리(GPU, TPU와 같은)에서 사용할 수 있습니다.
# 텐서는 불변성(immutable)을 가집니다.

#   2.1 Numpy 호환성 
#     텐서플로 연산은 자동으로 넘파이 배열을 텐서로 변환합니다.
#     넘파이 연산은 자동으로 텐서를 넘파이 배열로 변환합니다.
#     텐서는 .numpy() 메서드(method)를 호출하여 넘파이 배열로 변환할 수 있습니다. 
#     가능한 경우, tf.Tensor와 배열은 메모리 표현을 공유하기 때문에 이러한 변환은 일반적으로 간단(저렴)합니다. 
#     그러나 tf.Tensor는 GPU 메모리에 저장될 수 있고, 넘파이 배열은 항상 호스트 메모리에 저장되므로, 
#     이러한 변환이 항상 가능한 것은 아닙니다. 따라서 GPU에서 호스트 메모리로 복사가 필요합니다.
import numpy as np
ndarray = np.ones([3, 3])
print("텐서플로 연산은 자동적으로 넘파이 배열을 텐서로 변환합니다.")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print("그리고 넘파이 연산은 자동적으로 텐서를 넘파이 배열로 변환합니다.")
print(np.add(tensor, 1))
print(".numpy() 메서드는 텐서를 넘파이 배열로 변환합니다.")
print(tensor.numpy())

# 3. GPU 가속
# 대부분의 텐서플로 연산은 GPU를 사용하여 가속화됩니다. 어떠한 코드를 명시하지 않아도, 
# 텐서플로는 연산을 위해 CPU 또는 GPU를 사용할 것인지를 자동으로 결정합니다. 
# 필요시 텐서를 CPU와 GPU 메모리 사이에서 복사합니다. 
# 연산에 의해 생성된 텐서는 전형적으로 연산이 실행된 장치의 메모리에 의해 실행됩니다. 
x = tf.random.uniform([3, 3])
print("GPU 사용이 가능한가 : ", tf.test.is_gpu_available()),
print("텐서가 GPU #0에 있는가 : ", x.device.endswith('GPU:0'))

#   3.1 장치이름
#     Tensor.device 는 텐서를 구성하고 있는 호스트 장치의 풀네임을 제공합니다.
#     이러한 이름은 프로그램이 실행중인 호스트의 네트워크 주소 및 해당 호스트 내의 장치와 같은 많은 세부 정보를 인코딩하며, 
#     이것은 텐서플로 프로그램의 분산 실행에 필요합니다. 텐서가 호스트의 N번째 GPU에 놓여지면 문자열은 GPU:<N>으로 끝납니다. 
#   3.2 명시적 장치 배제
#     텐서플로에서 "배치(replacement)"는 개별 연산을 실행하기 위해 장치에 할당(배치)하는 것입니다. 
#     앞서 언급했듯이, 명시적 지침이 없을 경우 텐서플로는 연산을 실행하기 위한 장치를 자동으로 결정하고, 
#     필요시 텐서를 장치에 복사합니다. 그러나 텐서플로 연산은 tf.device 을 사용하여 특정한 장치에 명시적으로 배치할 수 있습니다. 
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

# 4. 데이터셋
#   4.1 소스 데이터셋 생성
'''
     1) numpy에서 불러 오기 
       x = np.random.sample((100,2)) 
       dataset = tf.data.Dataset.from_tensor_slices(x)
     2) tensor에서 불러 오기
       dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2])) 
     3) placeholder
       x = tf.placeholder(tf.float32, shape=[None,2])
       dataset = tf.data.Dataset.from_tensor_slices(x)
     4) generator
       sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])
       def generator():
     ​    for el in sequence:
           yield el
       dataset = tf.data.Dataset().batch(1).from_generator(generator,
 ​                                           output_types= tf.int64, 
                                            output_shapes=(tf.TensorShape([None, 1])))
     5) csv 파일에서 불러 오기
       dataset = tf.contrib.data.make_csv_dataset(CSV_PATH, batch_size=32)
     6) 파일로부터 읽어들이는 객체인 TextLineDataset 또는 TFRecordDataset 를 사용하여 소스 데이터셋을 생성
'''

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
# CSV 파일을 생성합니다.
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

# tf.data.TextLineDataset 클래스는 텍스트 파일로부터 라인을 하나씩 추출하는 데이터셋
ds_file = tf.data.TextLineDataset(filename)

#   4.2 변환 적용 : 맵(map), 배치(batch), 셔플(shuffle)과 같은 변환 함수를 사용하여 데이터셋의 레코드에 적용
#     map함수를 이용해서 데이터 셋의 각 멤버에 사용자 지정 함수를 적용 할 수 있습니다
#     shuffle을 사용하면 epoch마다 데이터 셋을 섞을 수 있습니다.



ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

#   4.3 반복 : tf.data.Dataset은 레코드 순회를 지원하는 반복가능한 객체입니다.
print('ds_tensors 요소:')
for x in ds_tensors:
  print(x)

print('\nds_file 요소:')
for x in ds_file:
  print(x)

