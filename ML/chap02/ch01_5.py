import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 5. 선형 회귀(최소제곱법)을 위한 wave 데이터셋. n_samples = 40
X, y = mglearn.datasets.make_wave(n_samples=400)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:5], y[:5])

# 1. k-최근접 이웃 알고리즘 : 회귀
mglearn.plots.plot_knn_regression(n_neighbors=1)
images.image.save_fig("1.5.Make_Wave_knn_regression_n_neighbors_1")  
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
images.image.save_fig("1.5.Make_Wave_knn_regression_n_neighbors_3")  
plt.show()

# wave 데이터셋을 훈련 세트와 테스트 세트로 나눕니다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

########################################################################
# 1. k-최근접 이웃 알고리즘 : 회귀
# 이웃의 수를 3으로 하여 모델의 객체를 만듭니다.
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=3)
# 훈련 데이터와 타깃을 사용하여 모델을 학습시킵니다.
reg.fit(X_train, y_train)
print("테스트 세트 예측:\n{}".format(reg.predict(X_test)))
print("테스트 세트 R^2: {:.2f}".format(reg.score(X_test, y_test)))

# n_neighbors 값에 따라 최근접 이웃 회귀로 만들어진 예측 회귀 비교
fig, axes = plt.subplots(1, 3, figsize=(15, 8))
# -3과 3 사이에 1,000개의 데이터 포인트를 만듭니다.
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9 이웃을 사용한 예측을 합니다.
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    # 훈련
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=2)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=2)
    ax.set_title("{} 이웃의 훈련 스코어: {:.2f} 테스트 스코어: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("특성")
    ax.set_ylabel("타깃")
axes[0].legend(["모델 예측", "훈련 데이터/타깃", "테스트 데이터/타깃"], loc="best")
images.image.save_fig("1.5.Make_Wave_knn_regression_n_neighbors_1_3_9")  
plt.show()

# n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 101)

for n_neighbors in neighbors_settings:
    # 모델 생성
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(reg.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(reg.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
images.image.save_fig("1.5.Make_Wave_knn_regression_n_neighbors_1_10")  
plt.show()


# KNeighbors 분류기에 중요한 매개변수는 두 개
# 1. 데이터 포인트 사이의 거리를 재는 방법
# 2. 이웃의 수
# 이웃의 수는 3개나 5개 정도로 적을 때 잘 작동하지만, 이 매개변수는 잘 조정해야 합니다. 
# 거리 재는 방법은 기본적으로 여러 환경에서 잘 동작하는 유클리디안 거리 방식을 사용합니다. 
# 
# k-NN의 장점은 
# 1. 이해하기 매우 쉬운 모델
# 2. 많이 조정하지 않아도 자주 좋은 성능을 발휘
# k-NN의 단점은 
# 1. 훈련 세트가 매우 크면 (특성의 수나 샘플의 수가 클 경우) 예측이 느려집니다. 
# 2. k-NN 알고리즘을 사용할 땐 데이터를 전처리하는 과정이 중요합니다(3장 참고). 
# 3. (수백 개 이상의) 많은 특성을 가진 데이터셋에는 잘 동작하지 않음
# 4. 특성 값 대부분이 0인 (즉 희소한) 데이터셋과는 특히 잘 작동하지 않음











