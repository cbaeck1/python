import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
print(sys.path)
import images.image

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# 1. 붓꽃iris 데이터셋
# iris_dataset의 키: dict_keys(['target_names', 'feature_names', 'DESCR', 'data', 'target'])
print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 크기: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))

print("target의 타입: {}".format(type(iris_dataset['target'])))
print("target의 크기: {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))

# 성과 측정: 훈련 데이터와 테스트 데이터
# 레이블된 데이터(150개의 붓꽃 데이터)를 두 그룹으로
# 75% 를 레이블 데이터와 함께 훈련 세트로 뽑습니다. 나머지 25%는 레이블 데이터와 함께 테스트 세트
# 여러 번 실행해도 결과가 똑같이 나오도록 유사 난수 생성기에 넣을 난수 초깃값을 random_state 매개변수로 전달
# train_test_split 함수의 반환값은 X_train, X_test, y_train, y_test이며 모두 NumPy 배열
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

########################################################################
# 1. k-최근접 이웃 알고리즘
# scikit-learn의 모든 머신러닝 모델은 Estimator라는 파이썬 클래스로 각각 구현
# k-최근접 이웃 분류 알고리즘은 neighbors 모듈 아래 KNeighborsClassifier 클래스에 구현
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)

# knn 객체는 훈련 데이터로 모델을 만들고 새로운 데이터 포인트에 대해 예측하는 알고리즘을 캡슐화한 것입니다. 
# 또한 알고리즘이 훈련 데이터로부터 추출한 정보를 담고 있습니다.
# KNeighborsClassifier 의 경우는 훈련 데이터 자체를 저장하고 있습니다

# 훈련 데이터인 NumPy 배열 X_train 과 훈련 데이터의 레이블을 담고 있는 NumPy 배열 y_train 을 매개변수
clf.fit(X_train, y_train)

# 예측하기 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = clf.predict(X_new)
print("예측: {}".format(prediction))
print("예측한 타깃의 이름: {}".format(iris_dataset['target_names'][prediction]))

# 모델 평가하기
y_pred = clf.predict(X_test)
print("테스트 세트에 대한 예측값:\n {}".format(y_pred))
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
print("테스트 세트의 정확도: {:.2f}".format(clf.score(X_test, y_test)))

# n_neighbors 값이 각기 다른 최근접 이웃 모델이 만든 결정 경계
# 이웃의 수를 늘릴수록 결정 경계는 더 부드러워집니다
# 2개의 특성으로
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메서드는 self 객체를 반환 객체 생성과 fit 메서드를 한 줄에 
    # plot_2d_separator 에서 그릴 X_train 갯수만큼만 fit 
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train[:,:2], y_train)
    mglearn.plots.plot_2d_separator(clf, X_train[:,:2], fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("sepal length")
    ax.set_ylabel("sepal width")
axes[0].legend(loc=3)
images.image.save_fig("1.1.Iris_KNN_n_neighbors_1_3_9", "ml")  
plt.show()

# n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 21)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
images.image.save_fig("1.1.Iris_KNN_n_neighbors_1_10", "ml")  
plt.show()
