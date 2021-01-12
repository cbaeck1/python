import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 3. 위스콘신 유방암 Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer라고 하겠습니다). 
# 각 종양은 양성benign(해롭지 않은 종양)과 악성malignant(암 종양)으로 레이블되어 있고, 
# 조직 데이터를 기반으로 종양이 악성인지를 예측할 수 있도록 학습하는 것이 과제
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer['DESCR']+ "\n...")
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))
print(cancer.data, cancer.target)
print(cancer.data[:,:2])

# 훈련/테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, random_state=0)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

from sklearn.svm import SVC
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("훈련 세트 정확도 : {:.2f}".format(svm.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.2f}".format(svm.score(X_test,y_test)))

# [MinMaxScaler 를 사용하여 데이터 전처리 후 성능]
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X_train)
X_train_scaled = minmax_scaler.transform(X_train)
X_test_scaled = minmax_scaler.transform(X_test)
print("X_train_scaled 크기: {}".format(X_train_scaled.shape))
print(X_train_scaled)
print("X_test_scaled 크기: {}".format(X_test_scaled.shape))

# 조정된 데이터로 SVM 학습
svm_scaled = SVC(C=100)
svm_scaled.fit(X_train_scaled, y_train)
print("스케일 조정(MinMaxScaler)된 훈련 세트 정확도 : {:.2f}".format(svm_scaled.score(X_train_scaled,y_train)))
print("스케일 조정(MinMaxScaler)된 테스트 세트 정확도 : {:.2f}".format(svm_scaled.score(X_test_scaled,y_test)))

# [StandardScaler 를 사용하여 데이터 전처리 후 성능]
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train_scaled_standard = standard_scaler.transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)
print("X_train_scaled_standard 크기: {}".format(X_train_scaled_standard.shape))
print(X_train_scaled_standard)
print("X_test_scaled_standard 크기: {}".format(X_test_scaled_standard.shape))

# 조정된 데이터로 SVM 학습
svm_scaled_standard = SVC(C=100)
svm_scaled_standard.fit(X_train_scaled_standard, y_train)
print("스케일 조정(StandardScaler)된 훈련 세트 정확도 : {:.2f}".format(svm_scaled_standard.score(X_train_scaled_standard,y_train)))
print("스케일 조정(StandardScaler)된 테스트 세트 정확도 : {:.2f}".format(svm_scaled_standard.score(X_test_scaled_standard,y_test)))




