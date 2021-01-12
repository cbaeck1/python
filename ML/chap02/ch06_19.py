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

# 18. # 클리블랜드(Cleveland) 심장병 재단에서 제공한 작은 데이터셋
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

########################################################################
# 6. 커널 서포트 벡터 머신 
# RBF 커널 SVM을 유방암 데이터셋에 적용해보겠습니다. 기본값 C=1, gamma=1/n_features를 사용

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))

# 과대 적합됬음을 알 수 있다.
# SVM은 매개변수와 데이터 스케일에 매우 민감하다
# 각 특성의 최대.최소값을 로그스케일로 표현

plt.boxplot(X_train, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("특성 목록")
plt.ylabel("특성 크기")
plt.title('heart 데이터셋의 특성 값 범위(y 축은 로그 스케일)')
images.image.save_fig("19.heart_rbf_Scatter")  
plt.show()

# SVM을 위한 데이터 전처리
# 커널 SVM에서는 모든 특성 값을 0과 1 사이로 맞추는 방법을 많이 사용

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면
# 각 특성에 대해 최솟값은 0, 최대값은 1입니다.
X_train_scaled = (X_train - min_on_training) / range_on_training
print("특성별 최소 값\n{}".format(X_train_scaled.min(axis=0)))
print("특성별 최대 값\n {}".format(X_train_scaled.max(axis=0)))

# 테스트 세트에도 같은 작업을 적용 : 훈련 세트에서 계산한 최솟값과 범위를 사용
X_test_scaled = (X_test - min_on_training) / range_on_training

#  
svc_scaled = SVC()
svc_scaled.fit(X_train_scaled, y_train)
print("훈련 세트 정확도: {:.3f}".format(svc_scaled.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc_scaled.score(X_test_scaled, y_test)))

# C 값 증가 -> 모델 성능 향상
svc_scaled_1000 = SVC(C=1000)
svc_scaled_1000.fit(X_train_scaled, y_train)
print("훈련 세트 정확도: {:.3f}".format(svc_scaled_1000.score(X_train_scaled, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc_scaled_1000.score(X_test_scaled, y_test)))


# < 장단점 >
# 데이터의 특성이 몇개 안되더라도 복잡한 결정 경계를 만들 수 있다.
# 저/고 차원이 데이터에서 모두 잘 작동하지만, 샘플이 많은 경우는 잘 맞지않는다.
# 데이터 전처리와 매개변수 석정에 주의해야한다. => 랜덤포레스트나 그래디언트 부스팅을 사용하는 이유

# < 매개변수 >
# 규제 매개변수 값인 C값이 클수록 모델 복잡도는 올라간다.
# RBF커널은 가우시안 커널 폭의 역수인 gamma 매개변수를 더 가진다.
# (SVM에는 RBF커널 말고도 다른 컬널이 많다.)

