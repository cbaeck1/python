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

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 유방암 데이터셋 커널SVM(SVC)를 적용하고 MinmaxScaler 를 이용
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#################################################################################
# 3. MinMaxScaler 의 fit 메서드는 훈련세트에 있는 특성마다 최대/최소값을 계산한다.
scaler.fit(X_train)
# MinMaxScaler(copy=True, feature_range=(0, 1))
# fit 메서드로 학습한 변환을 적용하려면 스케일 객체의 transform 메서드를 사용한다.
# 데이터 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 후 데이터셋의 속성을 출력
print("변환된 후 크기: {}".format(X_train_scaled.shape))
# 변환된 후 크기: (426, 30)
# 데이터 스케일을 변환해도 개수에는 변화 없다. 
print("스케일 조정 전 특성별 최소값: {}".format(X_train.min(axis=0)))
print("스케일 조정 전 특성별 최대값: {}".format(X_train.max(axis=0)))
print("스케일 조정 후 X_train 특성별 최소값: {}".format(X_train_scaled.min(axis=0)))
print("스케일 조정 후 X_train 특성별 최대값: {}".format(X_train_scaled.max(axis=0)))
print("스케일 조정 후 X_test 특성별 최소값: \n{}".format(X_test_scaled.min(axis=0)))
print("스케일 조정 후 X_test 특성별 최대값: \n{}".format(X_test_scaled.max(axis=0)))

#################################################################################
# 1. StandardScaler 를 사용
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)

X_train_scaled_standard = standard_scaler.transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)

########################################################################
# 6. 커널 서포트 벡터 머신 
# 데이터 전처리 효과를 성능 측정
from sklearn.svm import SVC
svm = SVC(C=100)

svm.fit(X_train, y_train)
print("훈련 세트 정확도 : {:.2f}".format(svm.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.2f}".format(svm.score(X_test, y_test)))

svm_scaled = SVC(C=100)
svm_scaled.fit(X_train_scaled, y_train)
print("스케일 조정(MinmaxScaler)된 훈련 세트 정확도 : {:.2f}".format(svm_scaled.score(X_train_scaled, y_train)))
print("스케일 조정(MinmaxScaler)된 테스트 세트 정확도 : {:.2f}".format(svm_scaled.score(X_test_scaled, y_test)))

svm_scaled_standard = SVC(C=100)
svm_scaled_standard.fit(X_train_scaled_standard, y_train)
print("스케일 조정(StandardScaler)된 훈련 세트 정확도 : {:.2f}".format(svm_scaled_standard.score(X_train_scaled_standard, y_train)))
print("스케일 조정(StandardScaler)된 테스트 세트 정확도 : {:.2f}".format(svm_scaled_standard.score(X_test_scaled_standard, y_test)))


