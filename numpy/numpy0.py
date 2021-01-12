'''
출처: https://rfriend.tistory.com/285?category=675917 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

1. 다차원 배열 ndarray 만들기
2. 무작위 표본 추출, 난수 만들기 (random sampling, random number generation)
(1-1) 이항분포로 부터 무작위 표본 추출 (Random sampling from Binomial Distribution) : np.random.binomial(n, p, size)
(1-2) 초기하분포에서 무작위 표보 추출 (Random sampling from Hypergeometric distribution) 
      np.random.hypergeometric(ngood, nbad, nsample, size)
(1-3) 포아송분포로 부터 무작위 표본 추출 : np.random.poisson(lam, size)
(2-1) 정규분포로부터 무작위 표본 추출 : np.random.normal(loc, scale, size)
(2-2) t-분포로 부터 무작위 표본 추출 : np.random.standard_t(df, size)
(2-3) 균등분포로 부터 무작위 표본 추출 : np.random.uniform(low, high, size)
이산형 균등분포에서 정수형 무작위 표본 추출 : np.random.randint(low, high, size)
(2-4) F-분포로 부터 무작위 표본 추출 : np.random.f(dfnum, dfden, size)
(2-5) 카이제곱분포로 부터 무작위 표본 추출 : np.random.chisquare(df, size)
      (Random sampling from Chisq-distribution)
3. ndarray 데이터 형태 지정 및 변경 (Data Types for ndarrays)
(1) 데이터 형태의 종류 (Data Types)
(2) NumPy 데이터 형태 지정해주기
(3) 데이터 형태 변환 (converting data type) : object.astype(np.Type)
(4) Python의 int와 NumPy의 int64 비교 
     (difference between Python's native int and NumPy's int64)

4. 배열과 배열, 배열과 스칼라 연산 (Numerical Operations between Arrarys and Scalars)
(1) 배열과 스칼라의 산술 연산 (Arithmetic operations with an array and scalar)
   * NumPy 와 Pure Python 속도 비교
(2) 같은 크기 배열 간 산술 연산
     (Arithmetic elementwise operstions between equal-size arrays)
(3) 배열 간 비교 연산 (Comparison operations between equal-size arrays)
(4) 배열 간 할당 연산 (Assignment operations between equal-size arrays)
(5) 배열 간 논리 연산 (Logical operations between equal-size arrays)
(6) 소속 여부 판단 연산 (Membership operators)
 배열의 차원, 크기가 다를 때
 => ValueError: operands could not be broadcast together with shapes (4,) (5,) 

5. 다른 차원의 배열 간 산술연산 시 Broadcasting
1) Broadcasting over axis 1 with a Scalar 
2) Broadcasting over axis 0 with a 1-D array
3) Broadcasting over axis 1 with a 2-D array
4) Broadcasting over axis 0 with a 3-D array

6. NumPy 배열에 축 추가하기 (adding axis to NumPy Array) : np.newaxis, np.tile
(1) indexing으로 길이가 1인 새로운 축을 추가하기 : arr(:, np.newaxis, :)
(2) 배열을 반복하면서 새로운 축을 추가하기 : np.tile(arr, reps)

7. 행렬의 행과 열 바꾸기, 축 바꾸기, 전치행렬 : a.T, np.transpose(a), np.swapaxes(a,0,1)
(1-1) Transposing 2 D array : a.T attribute
(1-2) Transposing 2D array : np.transpose() method
(1-3) Transposing 2D array : np.swapaxes() method
(2-1) Transposing 3D array : a.T attribute
(2-2) Transposing 3D array : np.transpose() method
(2-3) Transposing 3D array : np.swapaxes() method

8. 배열의 일부분 선택하기, indexing and slicing an ndarray
(1-1) Indexing a subset of 1D array : a[from : to]
(1-2) array slices are views of the original array and are not a copy
(1-3) indexing한 배열을 복사하기 : arr[0:5].copy()
(2-1) Indexing and Slicing 2D array with comma ',' : d[0:3, 1:3]
(2-2) Indexing and Slicing 2D array with square bracket '[ ][ ]' : d[0:3][1:3]
(2-3) array slices are views of the original array and are not a copy
(3-1) Indexing and Slicing of 3D array : e[0, 0, 0:3]
(3-2) 축 하나를 통째로 가져오기( indexing the entire axis by using colon ':')
Python NumPy 배열 Indexing과 R의 행렬 Indexing 비교

9. Boolean 조건문으로 배열 인덱싱 (Boolean Indexing)
(1) 특정 조건을 만족하는 배열의 모든 열을 선별하기 : ==
(2) 특정 조건을 만족하지 않는 배열의 모든 열을 선별하기 : !=, ~(==)
(3) 복수의 조건으로 배열의 특정 열 선별하기 : & (and), | (or)
(4) Booean 조건에 해당하는 배열 Indexing에 스칼라 값을 할당하기

10. 정수 배열을 사용해서 다차원 배열 인덱싱 하기 : Fancy Indexing
(1) 특정 순서로 다차원 배열의 행(row)을 Fancy Indexing 하기
(2) 특정 순서로 다차원 배열의 행(row)과 열(column)을 Fancy Indexing 하기
(3) Fancy Indexing은 view가 아니라 copy 를 생성

11. 범용 함수 (universal functions) : 
(1-1) 올림 혹은 내림 범용 함수 (round universal functions)
(1-2) 단일 배열 unary ufuncs : 합(sum), 누적합(cum_sum), 곱(product), 누적곱(cum_prod), 차분(difference), gradient 범용함수
(1-3) 지수함수(exponential function), 로그함수(logarithmic function)
(1-4) 삼각함수(trigonometric functions)
(1-5) 단일 배열 unary ufuncs : 절대값, 제곱근, 제곱값, 정수와 소수점 분리, 부호 함수
(1-6) 단일 배열 unary ufuncs : 논리형 함수(Logical function)

12. 절대값 함수 np.abs(x), 부호 판별 함수 np.sign(x)를 이용해서 특이값(Outlier) 찾고 다른 값으로 대체하기

13. reshape에서 -1 은 무슨 의미인가? (reshape(-1, 1))
(1) reshape(-1, 정수) 의 행(row) 위치에 '-1'이 들어있을 경우
(2) reshape(정수, -1) 의 열(column) 위치에 '-1'이 들어있을 경우
(3) reshape(-1) 인 경우
(4) ValueError: cannot reshape array of size 12 into shape (5)
(5) ValueError: can only specify one unknown dimension

14. 다차원 배열을 1차원 배열로 평평하게 펴주는 ravel(), flatten() 함수
배열을 옆으로, 위 아래로 붙이기 : np.r_, np.c_, np.hstack(), np.vstack(), np.column_stack(), np.concatenate(axis=0), np.concatenate(axis=1)
numpy 집합함수 (set functions)
numpy 최소, 최대, 조건 색인값 : np.argmin(), np.argmax(), np.where()
numpy array 정렬, 거꾸로 정렬, 다차원 배열 정렬
numpy 배열 외부 파일로 저장하기(save), 외부 파일을 배열로 불러오기(load)
numpy 배열을 여러개의 하위 배열로 분할하기 (split an array into sub-arrays)
선형대수 함수 (Linear Algebra)

배열에서 0보다 작은 수를 0으로 변환하는 방법
배열에 차원 추가하기 (Adding Dimensions to a Numpy Array)





'''