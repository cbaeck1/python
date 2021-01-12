import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image


# 3. 위스콘신 유방암Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer라고 하겠습니다). 
# 각 종양은 양성benign(해롭지 않은 종양)과 악성malignant(암 종양)으로 레이블되어 있고, 
# 조직 데이터를 기반으로 종양이 악성인지를 예측할 수 있도록 학습하는 것이 과제
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))
print(cancer.data, cancer.target)
print(cancer.data[:,:2])

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))

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
# 2개의 특성으로
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(cancer.data[:,:2], cancer.target)
    mglearn.plots.plot_2d_separator(clf, cancer.data[:,:2], fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(cancer.data[:, 0], cancer.data[:, 1], cancer.target, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend(loc=3)

images.image.save_fig("1.3.breast_cancer_KNN_n_neighbors_1_3_9")  
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
images.image.save_fig("1.3.breast_cancer_KNN_n_neighbors_1_10")  
plt.show()






