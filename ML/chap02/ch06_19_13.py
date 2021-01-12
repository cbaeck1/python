import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 19. 클리블랜드(Cleveland) 심장병 재단에서 제공한 작은 데이터셋
# 이 CSV 파일은 수백 개의 행으로 이루어져 있습니다. 
# 각 행은 환자 한 명을 나타내고 각 열은 환자에 대한 속성 값입니다. 
# 이 정보를 사용해 환자의 심장병 발병 여부를 예측
# 1.1 판다스로 데이터프레임 만들기
dataframe = pd.read_csv('data/heart.csv')
print(dataframe.head(), dataframe.shape)

# Categorical
X_dafaframe = dataframe.loc[:,'age':'thal']
X_dafaframe.thal = pd.Categorical(X_dafaframe.thal)
X_dafaframe['thal'] = X_dafaframe.thal.cat.codes
X_dafaframe = X_dafaframe.apply(pd.to_numeric)
print(X_dafaframe.head(), X_dafaframe.shape, X_dafaframe.info())

# numpy array
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X_dafaframe)
X_heart = imputer.transform(X_dafaframe)

Y_heart = dataframe['target']
print(Y_heart.head(), Y_heart.shape)

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, stratify=Y_heart, random_state=66)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))

#################################################################################
# 유방암 데이터셋 커널SVM(SVC)를 적용하고 MinmaxScaler 를 이용
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
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

#################################################################################
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


