import numpy as np

# 원소별 연산 : relu 함수와 덧셈
def naive_relu(x):
    assert len(x.shape) == 2    # x는 2D 넘파이 배열입니다.
    x = x.copy()                # 입력 텐서 자체를 바꾸지 않도록 복사합니다.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2     # x와 y는 2D 넘파이 배열입니다.
    assert x.shape == y.shape
    x = x.copy()                 # 입력 텐서 자체를 바꾸지 않도록 복사합니다.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

x = np.array([[5, 78, 2, 34, 0], 
              [6, 79, 3, 35, 1], 
              [7, 80, 4, 36, 2]]) 
y = np.array([[5, 78, 2, 34, 0], 
              [6, 79, 3, 35, 1], 
              [7, 80, 4, 36, 2]])     

z = x + y               # 원소별 덧셈
print("z = x + y:", z)
z = np.maximum(z, 0.)   # 원소별 relu 함수    
print("z = np.maximum(z, 0.):", z)

#  텐서 점곱
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2   # x는 2D 넘파이 배열입니다.
    assert len(y.shape) == 1   # y는 넘파이 벡터입니다.
    assert x.shape[1] == y.shape[0]
    x = x.copy()               # 입력 텐서 자체를 바꾸지 않도록 복사합니다.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x

x = np.random.random((64, 3, 32, 10)) # x는 (64, 3, 32, 10) 크기의 랜덤 4D 텐서입니다.
y = np.random.random((32, 10))        # y는 (32, 10) 크기의 랜덤 2D 텐서입니다.
z = np.maximum(x, y)                  # 출력 z 크기는 x와 동일하게 (64, 3, 32, 10)입니다.
print("z = np.maximum(x, y):", z.shape, len(z.shape))

a = np.array([0.5, 0.3, 0.1, 0.08, 0.02])  # a는 (5, ) 크기의 랜덤 1D 텐서입니다.
b = np.arange(15).reshape(5, 3)            # b는 (5, 3) 크기의 랜덤 2D 텐서입니다.
print(a.shape, b.shape, len(a.shape), len(b.shape))
print(np.dot(a, b))
z = np.dot(a, b)                           # z는 (3, ) 크기의 랜덤 1D 텐서입니다.
print("z = np.dot(a, b):", z)
# z = x · y
x = np.random.random((32, 10))        # x는 (32, 10) 크기의 랜덤 2D 텐서입니다.
y = np.random.random((10, 3))         # y는 (10, 3) 크기의 랜덤 2D 텐서입니다.
print(x.shape, y.shape, len(x.shape), len(y.shape))
z = np.dot(x, y)                      # z는 (32, 3) 크기의 랜덤 2D 텐서입니다.
print("z = np.dot(x, y):", z.shape)

def naive_vector_dot(x, y):
    assert len(x.shape) == 1  # x와 y는 넘파이 벡터입니다.
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2   # x는 넘파이 행렬입니다.
    assert len(y.shape) == 1   # y는 넘파이 벡터입니다.
    assert x.shape[1] == y.shape[0]  # x의 두 번째 차원이 y의 첫 번째 차원과 같아야 합니다!

    z = np.zeros(x.shape[0])   # 이 연산은 x의 행과 같은 크기의 0이 채워진 벡터를 만듭니다.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z

def naive_matrix_dot(x, y):
    assert len(x.shape) == 2   # x와 y는 넘파이 행렬입니다.
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]  # x의 두 번째 차원이 y의 첫 번째 차원과 같아야 합니다!

    z = np.zeros((x.shape[0], y.shape[1]))  # 이 연산은 0이 채워진 특정 크기의 벡터를 만듭니다.
    for i in range(x.shape[0]):     # x의 행을 반복합니다.
        for j in range(y.shape[1]): # y의 열을 반복합니다.
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

# 텐서 크기 변환
print('텐서 크기 변환')
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])

print(x.shape)
x = x.reshape((6, 1))
print(x)
x = x.reshape((2, 3))
print(x)


x = np.zeros((300, 20))  # 모두 0으로 채워진 (300, 20) 크기의 행렬을 만듭니다.
x = np.transpose(x)
print(x.shape)






