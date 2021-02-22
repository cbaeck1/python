import numpy as np

# np.array객체 
# 1. 동일한 데이터타입을 저장, 고정길이 (크기를 변경하면 새로 메모리에 할당되고 이전 값은 삭제)
#    연속적 메모리 공간
# 2. 벡터화 계산
#    브로드캐스트
# 3. 값 접근 : 원소하나, 행단위, 열단위

# 1. 동일한 데이터타입을 저장 
''' 
내부구조 
    1,2,3,4,5 을 할당
    1) Numpy Array (HEAD, data, dimensions, strides)
                          -> 1,2,3,4,5  (연속적인 물리적 메모리 공간)
    2) List (HEAD, lenths, items, Size, Reference Count, Object Type, Object Value)
                          -> Ox1,Ox3,Ox5,Ox6,Ox100 (주소)
                          -> 1,?,2,?,3,?,4,???...,5 ()
메모리저장방식
    1) little-endian : 저장을 뒤에서부터, 메모리 연산, 가산기 성능, 계산 성능, INTEL
    2) big-endian : 저장을 앞에서부터, 인간인식과 동일, 네트워크전송표준(tcp/ip,xns,sna), 비교 연산, 스택은 빠름, RISC
    3) not-relevant
sys.byteorder

itemsize : 메모리크기 (byte 단위)
dtype : 필드명과 데이터타입을 정의하고 직접 접근
shape : 배열의 형태
ndim : 차원수 = len(x.shape)
size : 원소의 개수
'''

# 0차원 배열 = 스칼라
a = np.array(10) # 10
print(a, a.ndim, a.shape, a.dtype, a.itemsize, a.nbytes, a.size)

# 1차원 배열 =  = [Column]
A = np.array( [1.0, 2.0] ) # 크기가 (2,) 인 1차원배열 : [ 갯수
# shape : numpy에서는 해당 array의 크기를 알 수 있다.
print(A, A.ndim, A.shape, A.dtype, A.itemsize, A.nbytes, A.size) # [ 1.  2.]

# 2차원 배열 = 행렬 = [Row][Column]
B = np.array( [[3,0],[0,6]] ) # 크기가 (2, 2) 인 2차원배열 : [[ 갯수
print(B, B.ndim, B.shape, B.dtype, B.itemsize, B.nbytes, B.size)
# [[1 2]
#  [3 4]]
# 3차원 배열 =  = [Layor][Row][Column]
X = np.array( [[[1,2],[3,4]], [[5,6],[7,8]]] ) # 크기가 (2, 2,2) 인 3차원배열 : [[[ 갯수
print(X, X.ndim, X.shape, X.dtype, X.itemsize, X.nbytes, X.size)

Y = np.array( [[[1,2],[3,4]], [[5,6],[7,8]], [[1,2],[3,4]]] ) # 크기가 (3, 2,2) 인 3차원배열 : [[[ 갯수
print(Y, Y.ndim, Y.shape, Y.dtype, Y.itemsize, Y.nbytes, Y.size)

Z = np.array( [[1,2,3], [1,2,3], [1,2,3]] ) # 크기가 (3, 3) 2차원배열 : [[ 갯수
print(Z, Z.ndim, Z.shape, Z.dtype, Z.itemsize, Z.nbytes, Z.size)

# 크기가 (3, 2,2,2) 인 4차원배열 : [[[[ 갯수
X4 = np.array( [ [[[1,2],[3,4]], [[5,6],[7,8]]], [[[11,12],[13,14]], [[15,16],[17,18]]], [[[21,22],[23,24]], [[25,26],[27,28]]] ] ) 
print(X4, X4.ndim, X4.shape, X4.dtype, X4.itemsize, X4.nbytes, X4.size)

# 2. 백터화 연산 : for문을 사용하지 않고 연산할 수 있음
print('===============백터화 연산')
# F = c * 8/5 + 32
c = np.array(np.arange(-100,101,1))
print(c)
f = c * 8/5 + 32
print(f)


# 연산 : 기본적으로 numpy에서 연산을 할때는 크기가 서로 동일한 array 끼리 연산이 진행된다.
#       이때 같은 위치에 있는 요소들 끼리 연산이 진행
# 브로드캐스트 : Shape이 다른 배열간 연산을 자동으로 지원 
# 1-2. 스칼라곱 (브로드캐스트)
print(A*10)
# [[10 20]
#  [30 40]]
# 1-3.  행렬의 형상 (행렬의 크기)
print(A.shape) # (2, 2)
# 1-4. 담긴 원소의 자료형 
print(A.dtype) # int32
# 1-5. 행렬의 덧셈 (브로드캐스트)
print(A+B)
# [[ 4  2]
#  [ 3 10]]
# 1-6. 행렬의 곱셈 (브로드캐스트)  : 각 원소별로 곱셈
print(A*B) 
# [[ 3  0]
#  [ 0 24]]

A = np.array([[1,2],[3,4]])
B = np.array([10,20])
# 2-1. 스칼라곱 (브로드캐스트)
print("===A*10===")
print(A*10)
# ===A*10===
# [[10 20]
#  [30 40]]
# 2-2 행렬의 곱셈 (브로드캐스트)  행,열의 위치에 맞는 원소끼리 곱하는 과정이다.
print("===A*B===")  
print(A*B)
# ===A*B===
# [[10 40]     1*10  2*20
#  [30 80]]    3*10  4*20
# 2-3 행렬의 나눗셈 (브로드캐스트)
print("===A/B===")  
print(A/B)
# [[0.1 0.1]     1/10  2/20
#  [0.3 0.2]]    3/10  4/20
# 2-4 행렬곱 : dot로 가능 (transpose)
print("==np.dot(A,B)===")  
print(np.dot(A, B))
# [ 50 110]  1*10+2*20    3*10+4*20

B = np.array([[10],[20]])
print(A*B)   
# [[10 20]      1*10  2*10
#  [60 80]]     3*20  4*20
print(A/B)
# [[0.1  0.2 ]  1/10  2/10
#  [0.15 0.2 ]] 3/20  4/20
print(np.dot(A, B))
# [[ 50]    1*10 + 2*20
#  [110]]   3*10 + 4*20


# 3. 원소값 찾기 : numpy에서 사용되는 인덱싱은 기본적으로 python 인덱싱과 동일
#   index 접근법 : 배열명[행][열]
#   slice 접근법 : 배열명[시작번호:끝번호] : 시작번호 포함, 끝번호 미포함
#   참조만 할당함으로 원본의 변경이 가능, 원본을 변경하지 않으려면 copy를 사용
#   __getitem__ , __setitem__
#   Boolean 접근법
#   
print('===============원소값 찾기')
A = np.array( [[1,2],[3,4],[5,6]] )
print("A = np.array( [[1,2],[3,4],[5,6]] )")  
# 1. 행 row 가져오기 -> 1차원배열
print("A[0,:]:", A[0,:]) # [1 2]
print("A[0]:", A[0]) # [1 2]
# 2. 원소 1개의 값 가져오기
print("A[0][1]:", A[0][1])
# 3. 열 column 가져오기  -> 1차원배열
print("A[:,0]:", A[:,0]) # [1 3 5]
print("A.__getitem__(0)", A.__getitem__(0)) # 행 row 가져오기 -> 1차원배열
# print("A.__getitem__(0,1)", A.__getitem__(0,1))
ar1 = np.array( [[True,False],[True,False],[True,False]] )
print("A.__getitem__(ar1)", A.__getitem__(ar1))  # [1 3 5]   1차원배열
ar2 = np.array( [[True,False],[True,True],[True,False]] )
print("A.__getitem__(ar2)", A.__getitem__(ar2))  # [1 3 4 5]  1차원배열
 
# [startindex:endindex:stepsize]
# 4. for 문
for row in A:
    print(row)
# [1 2]
# [3 4]
# [5 6]    

print('===============논리연산')
# 5. 1차원 배열로 변환
B = A.flatten()
print(B) # [1 2 3 4 5 6]
# 6. index를 배열로 주어 여러 원소를 접근
print(B[np.array([0,2,4])]) # [1 3 5]
# 6-1. 부등호 연산자를 통한 bool배열
print(B>2) # [False False  True  True  True  True]
# 6-2. 논리연산 : 배열명[논리연산]
print(B[B>2]) # [3 4 5 6]
print(B[B.nonzero()]) # [1 2 3 4 5 6]
# 6-3. 배열변경 : 배열명[논리연산] = 값
B[B>2] = 99
print(B) # [ 1  2 99 99 99 99]
# 원소가 2인 행의 데이터만 꺼내기
print(B[B==2]) # [2]
# 


# 초기화
np.zeros((2,3))
np.ones((2,3), dtype='int32')
np.full((2,3), 99, dtype='int32')
np.full_like(A.shape, 4)
np.empty((3,3))
np.identity(3)

# np.random.randn()는 기대값이 0이고, 표준편차가 1인 가우시안 정규 분포를 따르는 난수를 발생시키는 함수
# np.random.rand() : 0~1의 난수를 발생
np.random.rand(4,2)
np.random.random_sample(A.shape)
np.random.randint(-4,4, size=(3,3))

# 함수
arr1 = np.random.randn(5,3)
# 각 성분의 절대값 계산하기
np.abs(arr1)
# 각 성분의 제곱근 계산하기 ( == array ** 0.5)
np.sqrt(arr1)
# 각 성분의 제곱 계산하기
np.square(arr1)
# 각 성분을 무리수 e의 지수로 삼은 값을 계산하기
np.exp(arr1)
# 각 성분을 자연로그, 상용로그, 밑이 2인 로그를 씌운 값을 계산하기
np.log(arr1)
np.log10(arr1)
np.log2(arr1)
# 각 성분의 부호 계산하기(+인 경우 1, -인 경우 -1, 0인 경우 0)
np.sign(arr1)
# 각 성분의 소수 첫 번째 자리에서 올림한 값을 계산하기
np.ceil(arr1)
# 각 성분의 소수 첫 번째 자리에서 내림한 값을 계산하기
np.floor(arr1)
# 각 성분이 NaN인 경우 True를, 아닌 경우 False를 반환하기
np.isnan(arr1)
np.isnan(np.log(arr1))
# 각 성분이 무한대인 경우 True를, 아닌 경우 False를 반환하기
np.isinf(arr1)
# 각 성분에 대해 삼각함수 값을 계산하기(cos, cosh, sin, sinh, tan, tanh)
np.cos(arr1)
arr2 = np.random.randn(5,3)
# 두 개의 array에 대해 동일한 위치의 성분끼리 비교하여 최대값 또는 최소값 계산하기(maximum, minimum)
np.maximum(arr1,arr2)
# 두 개의 array에 대해 동일한 위치의 성분끼리 연산 값을 계산하기(add, subtract, multiply, divide)
np.multiply(arr1,arr2)
# 전체 성분의 합을 계산
np.sum(arr1)
# 열 간의 합을 계산
np.sum(arr1, axis=1)
# 행 간의 합을 계산
np.sum(arr1, axis=0)
# 전체 성분의 평균을 계산
np.mean(arr1)
# 행 간의 평균을 계산
np.mean(arr1, axis=0)
# 전체 성분의 표준편차, 분산, 최소값, 최대값 계산(std, var, min, max)
np.std(arr1)
np.min(arr1, axis=1)
# 전체 성분의 최소값, 최대값이 위치한 인덱스를 반환(argmin, argmax)
np.argmin(arr1)
np.argmax(arr1,axis=0)
# 맨 처음 성분부터 각 성분까지의 누적합 또는 누적곱을 계산(cumsum, cumprod)
np.cumsum(arr1)
np.cumsum(arr1,axis=1)
np.cumprod(arr1)
# 전체 성분에 대해서 내림차순으로 정렬
np.sort(arr1)[::-1]
# 행 방향으로 오름차순으로 정렬
np.sort(arr1,axis=0)

# Repeat
print('===============Repeat')
arr = np.array([[1,2,3]]) # 2차원
r1 = np.repeat(arr, 3, axis=0)
print(r1)
r2 = np.repeat(arr, 3, axis=1)
print(r2)

# copy
a = np.array( [1,2,3])
b = a
b[0] = 100
print(a)

b = a.copy()
b[0] = 100
print(a)

# 선형대수
print('==========선형대수==========')
# scalar는 number, Vector는 number들의 list, matrix는 number들의 array
scalar = 32
vecotor = [2, -8, 7]  # 크기와 방향을 표시
matrix = [[2, -8, 7], [1, -7, 6]]
print(scalar, vecotor, matrix)
print(type(scalar), type(vecotor), type(matrix))

# 벡터의 크기 = 거리 = x축의 변화, y축의 변화 = |A|
# 단위벡터는 크기가 1인 벡터
# 벡터의 정규화 = 벡터을 0 ~ 1 사이의 값으로 정규화
# 연산 : +, -, 스칼라곱, 곱(multiply) ==> ndarray로 계산
# 내적 : inner product, dot, n차원, A.B = |A|*|B|*cos(AB) = Ax*Bx + Ay*By
A = np.array( [[1,2]] )  # 2차원배열
B = np.array( [[3,4]] )  # 2차원배열
print(A+B, A.ndim, B.ndim, A.shape, B.shape)
print(A-B)
X = np.matrix( [1,2] )   # 2차원배열
Y = np.matrix( [3,4] )   # 2차원배열
print(X+Y, X.ndim, Y.ndim, X.shape, Y.shape)
print(X-Y)

A = np.array( [1,2] )    # 1차원배열
B = np.array( [3,4] )
print("==np.dot(A, B)  벡터의 내적===")  
print(np.dot(A, B), np.dot(A, B).shape)
# 외적 : vector product, cross product,  오른손의 법칙 방향, AXB = |A|*|B|*sin(AB) = a1*b2 - a2*b1 (2차원 -> 스칼라)
# A = [[a1, a2, a3]]   2차원배열 (1, 3)
# B = [[b1, b2, b3]]   2차원배열 (1, 3)
# AXB = (a2*b3-a3*b2, a3*b1-a1*b2, a1*b2-a2*b3)  (3차원 -> 3차원)
A = np.array( [[0, 0, 1]] )
B = np.array( [[0, 1, 0]] )
print("==np.cross(A, B)  벡터의 외적===")  
print(np.cross(A, B), np.cross(A, B).shape)
# inner 계산
# np.inner(A, B) = a1*b1 + a2*b2 + a3*b3
print("==np.inner(A, B)===")  
print(np.inner(A, B), np.inner(A, B).shape)
# outer 계산 : 첫번째 벡터의 전치와 두번째 벡터와의 dot연산
# np.outer(A, B) = [[a1*b1,a1*b2,a1*b3], [a2*b1,a2*b2,a2*b3], [a3*b1,a3*b2,a3*b3]]
print("==np.outer(A, B)===")  
print(np.outer(A, B), np.outer(A, B).shape)

# 대각행렬 : 주대각선 위에 있는 원소외의 원소가 모두 0인 행렬
# 항등행렬 : 주대각선 위에 있는 원소만 1인 행렬, 모든행렬과 내적연산할때 자기 자신이 되게하는 단위행렬
# 삼각행렬 : 상삼각행렬 + 하삼각행렬
# 상삼각행렬 : 주대각선 위에 있는 원소와 대각선 위에 있는 원소가 모두 0이 아닌 행렬
# 행렬의 연산 : +, -, 스칼라곱, 곱(multiply) ==> ndarray로 계산
# 행렬의 전치 : 역행렬 => transpose 함수
# 내적 : inner product, dot, n차원, A.B = |A|*|B|*cos(AB) = Ax*Bx + Ay*By
# np.inner(A, B) = a1*b1 + a2*b2 + a3*b3
# 행렬식 (det) : 정방행렬에 하나의 수를 대응 => 연립방정식의 해
# det(A) = a11*a22 - a21*a12
# 소행렬식 : i번째 행, j번째 열을 제거한 부분행렬의 행렬식, Mij
# 여인수 : 소행렬식을 이용한 값을 여인수로 표시 : C = (-1)^(i+j)Mij
# 여인수행렬 : 여인수로 구성된 행렬
# 수반행렬 : 여인수행렬의 전치행렬
# 역행렬 : 수반행렬을 행렬식으로 나눈 행렬
# outer 계산 : 첫번째 벡터의 전치와 두번째 벡터와의 dot연산
# 외적 : vector product, cross product,  오른손의 법칙 방향, AXB = |A|*|B|*sin(AB) = a1*b2 - a2*b1 (2차원 -> 스칼라)
# 대각행렬의 합 : 주대각선 위에 있는 원소외의 원소를 모두 더한 값 ==> trace 함수

# 행렬 : 벡터공간에서 벡터공간의 함수중 덧셈과 상수배가 보존되는 함수

# Reorganize
print('===============Reorganize')
before = np.array( [[1,2,3,4], [5,6,7,8]])
print(before)
after = before.reshape(4,2) # 또는 after = np.reshape(before, (4,2))
print("before.reshape(4,2)", after.shape)
print(after)
after = before.reshape(2,2,2)
print("before.reshape(2,2,2)", after.shape)
print(after)
# Exception has occurred: ValueError cannot reshape array of size 8 into shape (3,4)
# after = before.reshape(3,4)

np.arange(1,2, 0.1)
np.arange(10)   # start, step 생략가능. 정수로 생성
np.arange(10.)  # start, step 생략가능. 실수로 생성

np.linspace(0.,20.,11) # array([  0.,   2.,   4., ...,  16.,  18.,  20.])
np.eye(3) 
'''
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
'''

# astype() 메쏘드를 사용하면 배열에서 dtype을 바꿀 수 있다.
a.astype(int)  # a.astype('int34') 와 동일








