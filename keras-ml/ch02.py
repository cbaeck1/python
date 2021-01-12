
import keras
import matplotlib.pyplot as plt
import image

# train_images와 train_labels가 모델이 학습해야 할 훈련 세트를 구성합니다. 
# 모델은 test_images와 test_labels로 구성된 테스트 세트에서 테스트될 것입니다. 
# 이미지는 넘파이 배열로 인코딩되어 있고 레이블은 0에서부터 9까지의 숫자 배열입니다. 
# 이미지와 레이블은 일대일 관계를 가집니다.
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_images))
print(train_labels)

print(train_images.shape)
print(len(train_images))
print(train_labels)

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
image.save_fig("2.1 train_images")  
plt.show()

# 작업 순서
# 1. 훈련 데이터 train_images 와 train_labels 를 network 에 주입
# 2. network 는 이미지와 레이블을 연관시킬 수 있도록 학습
# 3. test_images 에 대한 예측을 network 에게 요청
# 4. 예측이 test_labels 와 맞는지 확인

from keras import models
from keras import layers
# 1. 훈련 데이터 train_images 와 train_labels 를 network 에 주입
#   1) 데이터정제필터
#   2) 컴파일 (손실함수, 옵티마이저, 지표)
#   3) 스케일 조정
#   4) 레이블 범주형
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 신경망의 핵심구성요소는 일종의 데이터정제필터라고 생각할 수 있는 층입니다. 
# 데이터가 들어가면 더 유용한 형태로 출력될 수 있도록 더 의미 있는 표현을 입력된 데이터로부터 추출합니다. 
# 대부분의 딥러닝은 간단한 층을 연결하여 구성되어 있고, 점진적으로 데이터를 정제하는 형태를 띠고 있습니다. 
# 딥러닝 모델은 데이터정제필터(층)가 연속되어 있는 데이터 프로세싱을 위한 여과기와 같습니다.
# 위 예에서는 완전 연결된 신경망 층인 Dense 층 2개가 연속되어 있습니다. 
# 두 번째층은 10개의 확률 점수가 들어 있는 배열(모두 더하면 1입니다)을 반환하는 softmax 층입니다. 
# 각 점수는 현재 숫자 이미지가 10개의 숫자 클래스 중 하나에 속할 확률입니다.
# 신경망이 훈련 준비를 마치기 위해서 컴파일 단계에 포함될 세 가지가 더 필요합니다:
#   1. 손실 함수 : 훈련 데이터에서 신경망의 성능을 측정하는 방법으로 네트워크가 옳은 방향으로 학습될 수 있도록 도와 줍니다.
#   2. 옵티마이저: 입력된 데이터와 손실 함수를 기반으로 네트워크를 업데이트하는 메커니즘입니다.
#   3. 훈련과 테스트 과정을 모니터링할 지표 : 여기에서는 정확도(정확히 분류된 이미지의 비율)만 고려하겠습니다.
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 훈련을 시작하기 전에 데이터를 네트워크에 맞는 크기로 바꾸고 모든 값을 0과 1 사이로 스케일을 조정합니다. 
# 앞서 우리의 훈련 이미지는 [0, 255] 사이의 값인 uint8 타입의 (60000, 28, 28) 크기를 가진 배열로 저장되어 있습니다. 
# 이 데이터를 0과 1 사이의 값을 가지는 float32 타입의 (60000, 28 * 28) 크기의 배열로 바꿉니다.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 레이블을 범주형으로 인코딩해야 합니다
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 2. network 는 이미지와 레이블을 연관시킬 수 있도록 학습
# 네트워크가 128개 샘플씩 미니 배치로 훈련 데이터를 다섯 번 반복
# 각 반복마다 네트워크가 배치에서 손실에 대한 가중치의 그래디언트를 계산하고 그에 맞추어 가중치를 업데이트
# 다섯 번의 epoch 동안 네트워크는 2,345번의 그래디언트 업데이트를 수행
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 3. test_images 에 대한 예측을 network 에게 요청
test_loss, test_acc = network.evaluate(test_images, test_labels)
# 4. 예측이 test_labels 와 맞는지 확인
print('test_acc:', test_acc)

