'''
신경망과의 첫 만남
이 노트북은 케라스 창시자에게 배우는 딥러닝 책의 2장 1절의 코드 예제입니다. 책에는 더 많은 내용과 그림이 있습니다. 
이 노트북에는 소스 코드에 관련된 설명만 포함합니다. 이 노트북의 설명은 케라스 버전 2.2.2에 맞추어져 있습니다. 
케라스 최신 버전이 릴리스되면 노트북을 다시 테스트하기 때문에 설명과 코드의 결과가 조금 다를 수 있습니다.
케라스 파이썬 라이브러리를 사용하여 손글씨 숫자 분류를 학습하는 구체적인 신경망 예제를 살펴보겠습니다. 
케라스나 비슷한 라이브러리를 사용한 경험이 없다면 당장은 이 첫 번째 예제를 모두 이해하지 못할 것입니다. 
아직 케라스를 설치하지 않았을지도 모릅니다. 괜찮습니다. 다음 장에서 이 예제를 하나하나 자세히 설명합니다. 
코드가 좀 이상하거나 요술처럼 보이더라도 너무 걱정하지 마세요. 일단 시작해 보겠습니다.
여기에서 풀려고 하는 문제는 흑백 손글씨 숫자 이미지(28x28 픽셀)를 10개의 범주(0에서 9까지)로 분류하는 것입니다. 
머신 러닝 커뮤니티에서 고전으로 취급받는 데이터셋인 MNIST를 사용하겠습니다. 
이 데이터셋은 머신 러닝의 역사만큼 오래되었고 많은 연구에 사용되었습니다. 
이 데이터셋은 1980년대에 미국 국립표준기술연구소에서 수집한 6만 개의 훈련 이미지와 1만 개의 테스트 이미지로 구성되어 있습니다. 
MNIST 문제를 알고리즘이 제대로 작동하는지 확인하기 위한 딥러닝계의 ‘Hello World’라고 생각해도 됩니다. 
머신 러닝 기술자가 되기까지 연구 논문이나 블로그 포스트 등에서 MNIST를 보고 또 보게 될 것입니다.
MNIST 데이터셋은 넘파이 배열 형태로 케라스에 이미 포함되어 있습니다:

2.1 신경망의 수학적 구성 요소
손실 함수loss function: 훈련 데이터에서 신경망의 성능을 측정하는 방법으로 네트워크가 옳은 방향으로 학습될 수 있도록 도와줍니다.
옵티마이저optimizer: 입력된 데이터와 손실 함수를 기반으로 네트워크를 업데이트하는 메커니즘입니다.
훈련과 테스트 과정을 모니터링할 지표: 정확도(정확히 분류된 이미지의 비율) 등

2.2 신경망을 위한 데이터 표현
1. 스칼라(0D 텐서)
하나의 숫자만 담고 있는 텐서를 scalar(scalar 텐서, 0차원 텐서, 0D 텐서)라고 부릅니다.
ndim 속성을 사용하여 축 개수를 확인
스칼라의 축 개수는 0 입니다
x = np.array(12)
print(x.ndim, x)

2. 벡터(1D 텐서)
숫자의 배열을 벡터 vector(1D 텐서) 라고 부릅니다.
벡터는 1개의 축
x = np.array([12, 3, 6, 14, 7]) 
print(x.ndim, x)

3. 행렬(2D 텐서)
벡터의 배열을 행렬 matrix(2D 텐서) 라고 부릅니다.
행렬에는 2개의 축
첫 번째 축에 놓여 있는 원소를 행, 두 번째 축에 놓여 있는 원소를 열
x = np.array([[5, 78, 2, 34, 0], 
              [6, 79, 3, 35, 1], 
              [7, 80, 4, 36, 2]]) 
print(x.ndim, x)

4. 3D 텐서와 고차원 텐서
2D 텐서의 배열을 3D 텐서

x = np.array([[[5, 78, 2, 34, 0], 
                [6, 79, 3, 35, 1], 
                [7, 80, 4, 36, 2]], 
               [[5, 78, 2, 34, 0], 
                [6, 79, 3, 35, 1], 
                [7, 80, 4, 36, 2]], 
               [[5, 78, 2, 34, 0], 
                [6, 79, 3, 35, 1], 
                [7, 80, 4, 36, 2]]]) 
print(x.ndim, x)

nD 텐서의 배열 n+1 텐서

5. 핵심 속성
축의 개수(rank)
크기shape
데이터 타입

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

6. 텐서의 실제 사례
1) 벡터 데이터: (samples, features) 크기의 2D 텐서
2) 시계열 데이터 또는 시퀀스sequence 데이터: (samples, timesteps, features) 크기의 3D 텐서
3) 이미지: (samples, height, width, channels) 또는 (samples, channels, height, width) 크기의 4D 텐서
4) 동영상: (samples, frames, height, width, channels) 또는 (samples, frames, channels, height, width) 크기의 5D 텐서

2.3 신경망의 톱니바퀴: 텐서 연산
1. 원소별 연산 : relu 함수와 덧셈 등 텐서에 있는 각 원소에 독립적으로 적용하여 병렬 구현
BLAS는 고도로 병렬화되고 효율적인 저수준의 텐서 조작 루틴이며, 전형적으로 포트란Fortran이나 C 언어로 구현되어 있고
넘파이는 시스템에 설치된 BLAS에 복잡한 일들을 위임
2. 브로드캐스팅
1) 큰 텐서의 ndim 에 맞도록 작은 텐서에 (브로드캐스팅 축이라고 부르는) 축이 추가됩니다.
2) 작은 텐서가 새 축을 따라서 큰 텐서의 크기에 맞도록 반복됩니다.
예) X의 크기는 (32, 10)이고 y의 크기는 (10,)
먼저 y에 비어 있는 첫 번째 축을 추가하여 크기를 (1, 10)으로 만듭니다. 
그런 다음 y를 이 축에 32번 반복하면 텐서 Y의 크기는 (32, 10)이 됩니다. 
여기에서 Y[i, :] == y for i in range(0, 32)입니다. 
이제 X와 Y의 크기가 같으므로 더할 수 있습니다.

구현 입장에서는 새로운 텐서가 만들어지면 매우 비효율적이므로 어떤 2D 텐서도 만들어지지 않습니다. 
반복된 연산은 완전히 가상적입니다. 이 과정은 메모리 수준이 아니라 알고리즘 수준에서 일어납니다
3. 텐서 점곱
x의 열과 y의 행 사이 벡터 점곱

4. 텐서 크기 변환
텐서의 크기를 변환한다는 것은 특정 크기에 맞게 열과 행을 재배열한다
크기가 변환된 텐서는 원래 텐서와 원소 개수가 동일
자주 사용하는 특별한 크기 변환은 전치transposition입니다. 행렬의 전치는 행과 열을 바꾸는 것을 의미

5. 텐서 연산의 기하학적 해석
모든 텐서 연산은 기하학적 해석이 가능
아핀 변환affine transformation 23, 회전, 스케일링scaling 등처럼 기본적인 기하학적 연산은 텐서 연산으로 표현

6. 딥러닝의 기하학적 해석
단순한 단계들이 길게 이어져 구현된 신경망을 고차원 공간에서 매우 복잡한 기하학적 변환을 하는 것으로 해석
예) 하나는 빨간색이고 다른 하나는 파란색인 2개의 색종이가 있다고 가정
    두 장을 겹친 다음 뭉쳐서 작은 공으로 만듭니다. 
    이 종이 공이 입력 데이터고 색종이는 분류 문제의 데이터 클래스입니다. 
    신경망(또는 다른 머신 러닝 알고리즘)이 해야 할 일은 종이 공을 펼쳐서 
    두 클래스가 다시 깔끔하게 분리되는 변환을 찾는 것입니다. 
    손가락으로 종이 공을 조금씩 펼치는 것처럼 딥러닝을 사용하여 
    3D 공간에서 간단한 변환들을 연결해서 이를 구현한다.

2.4 신경망의 엔진: 그래디언트 기반 최적화
2.4.1 변화율이란?
2.4.2 텐서 연산의 변화율: 그래디언트
2.4.3 확률적 경사 하강법
2.4.4 변화율 연결: 역전파 알고리즘


# 영화 리뷰 분류: 이진 분류 예제
# 뉴스 기사 분류: 다중 분류 문제
# 주택 가격 예측: 회귀 문제

# 과대적합과 과소적합




'''