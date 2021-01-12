import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

from IPython.display import display 
import graphviz

# 3. 위스콘신 유방암 Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer 라고 하겠습니다). 
# 각 종양은 양성 benign (해롭지 않은 종양)과 악성 malignant (암 종양)으로 레이블되어 있고, 
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

print("유방암 데이터의 특성별 최댓값:\n{}".format(cancer.data.max(axis=0)))

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
      stratify=cancer.target, random_state=0)
print("X_train 크기: {} {}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {} {}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {} {}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {} {}".format(y_test.shape, y_test.dtype))

########################################################################
# 7. 딥러닝
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("훈련 세트 정확도: {:.2f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(mlp.score(X_test, y_test)))

# 신경망도 데이터 스케일에 예민
# 모든 입력 특성을 평균0, 분산1에 변형
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

# 표준화 : 데이터에서 평균을 빼고 표준 편차로 나누면 평균 0, 표준 편차 1인 데이터로 변환  
# StandardScaler
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# 
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

#
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("훈련 세트 정확도: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

#
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("은닉 유닛")
plt.ylabel("입력 특성")
plt.colorbar()
plt.title('유방암 데이터셋으로 학습시킨 신경망의 첫 번째 층의 가중치 히트맵')
images.image.save_fig("3.cancer_heat_map_Scatter")  
plt.show()

# 장단점
# 대량의 데이터에 내재된 정보를 잡아내고 매우 복잡한 모델을 만들 수 있다.
# 충분한 연산 시간과 데이터를 주고 메개변수를 잘 조정하면 뛰어난 성능을 낸다.
# 학습시간이 오래 걸릴수 있다.
# 데이터 전처리 작업에 주의해야한다.

# 신경망에서 가장 중요한 매개변수
# 1. 은닉층의 개수
# 2. 각 은닉층의 유닛 수


