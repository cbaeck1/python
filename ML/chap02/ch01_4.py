import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격 Boston Housing 데이터셋
# 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 1970년대 보스턴 주변의 주택 평균 가격을 예측
# 이 데이터셋에는 데이터 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['DESCR']+ "\n...")
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.data[:,:2])
print("boston.data 타입: {}".format(type(boston.data)))
print("boston.target 타입: {}".format(type(boston.target)))

# Exception has occurred: ValueError Unknown label type: 'continuous'
# target을 세개의 값으로 변경
y_bin = np.array([0 if i < 17.0 else (1 if i < 25.0 else 2) for i in boston.target])

# 훈련 세트, 테스트 세트 random_state=66
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, y_bin, random_state=66)
print("X_train 타입: {}".format(type(X_train)))
print("y_train 타입: {}".format(type(y_train)))
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

########################################################################
# 1. k-최근접 이웃 알고리즘 : 분류 

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
# 훈련 세트를 사용하여 분류 모델을 학습
clf.fit(X_train, y_train)
print("훈련 세트 정확도: {:.2f}".format(clf.score(X_train, y_train)))
# 예측
prediction = clf.predict(X_test)
print("테스트 세트 예측: {}".format(prediction))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))

# n_neighbors 값이 각기 다른 최근접 이웃 모델이 만든 결정 경계
# 이웃의 수를 늘릴수록 결정 경계는 더 부드러워집니다
# 2개 특성만으로 
fig, axes = plt.subplots(1, 4, figsize=(10, 4))
for n_neighbors, ax in zip([1, 3, 6, 9], axes):
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train[:,:2], y_train)
    mglearn.plots.plot_2d_separator(clf, X_train[:,:2], fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend(loc=4)
images.image.save_fig("1.4.etended_boston_KNN_n_neighbors_1_3_9")  
plt.show()


# n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 11)

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
images.image.save_fig("1.4.etended_boston_KNN_n_neighbors_1_10")  
plt.show()









