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
plt.yticks(range(13), X_dafaframe.columns)
plt.xlabel("은닉 유닛")
plt.ylabel("입력 특성")
plt.colorbar()
plt.title('heart 데이터셋으로 학습시킨 신경망의 첫 번째 층의 가중치 히트맵')
images.image.save_fig("19.heart_heat_map_Scatter")  
plt.show()

# 장단점
# 대량의 데이터에 내재된 정보를 잡아내고 매우 복잡한 모델을 만들 수 있다.
# 충분한 연산 시간과 데이터를 주고 메개변수를 잘 조정하면 뛰어난 성능을 낸다.
# 학습시간이 오래 걸릴수 있다.
# 데이터 전처리 작업에 주의해야한다.

# 신경망에서 가장 중요한 매개변수
# 1. 은닉층의 개수
# 2. 각 은닉층의 유닛 수


