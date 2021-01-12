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
# 5. 결정트리 앙상블 : 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(forest.score(X_test, y_test)))

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
images.image.plot_feature_importances(forest, X_dafaframe, X_dafaframe.columns)
plt.title('heart 데이터로 만든 랜덤 포레스트 모델의 특성 중요도')
images.image.save_fig("19.heart_random_forest_feature_importances")  
plt.show()

# 랜덤 포레스트는 텍스트 데이터 같이 매우 차원이 높고 희소한 데이터에는 잘 작동하지 않습니다. 
# 이런 데이터에는 선형 모델이 더 적합합니다. 랜덤 포레스트는 매우 큰 데이터셋에도 잘 작동하며 
# 훈련은 여러 CPU 코어로 간단하게 병렬화할 수 있습니다. 
# 하지만 랜덤 포레스트는 선형 모델보다 많은 메모리를 사용하며 훈련과 예측이 느립니다. 
# 속도와 메모리 사용에 제약이 있는 애플리케이션이라면 선형 모델이 적합할 수 있습니다.

# n_estimators는 클수록 좋습니다. 더 많은 트리를 평균하면 과대적합을 줄여 더 안정적인 모델을 만듭니다. 
# 하지만 이로 인해 잃는 것도 있는데, 더 많은 트리는 더 많은 메모리와 긴 훈련 시간으로 이어집니다
# max_features는 각 트리가 얼마나 무작위가 될지를 결정하며 작은 max_features는 과대적합을 줄여줍니다. 
# 일반적으로 기본값을 쓰는 것이 좋은 방법입니다. 분류는 max_features=sqrt(n_features)이고 회귀는 max_features=n_features 입니다. 
# max_features나 max_leaf_nodes 매개변수를 추가하면 가끔 성능이 향상되기도 합니다. 
# 또 훈련과 예측에 필요한 메모리와 시간을 많이 줄일 수도 있습니다.

# 5. 결정트리 앙상블 : 그래디언트 부스팅 회귀 트리
# 여러 개의 결정 트리를 묶어 강력한 모델을 만드는 또 다른 앙상블 
# 기본적으로 그래디언트 부스팅 회귀 트리에는 무작위성이 없습니다. 대신 강력한 사전 가지치기가 사용
# 보통 하나에서 다섯 정도의 깊지 않은 트리를 사용하므로 메모리를 적게 사용하고 예측도 빠릅니다.
# 얕은 트리 같은 간단한 모델(약한 학습기weak learner라고도 합니다)을 많이 연결
# 랜덤 포레스트보다는 매개변수 설정에 조금 더 민감하지만 잘 조정하면 더 높은 정확도를 제공
# 중요한 매개변수는 이전 트리의 오차를 얼마나 강하게 보정할 것인지를 제어하는 learning_rate입니다. 
# 학습률이 크면 트리는 보정을 강하게 하기 때문에 복잡한 모델을 만듭니다. 
# n_estimators 값을 키우면 앙상블에 트리가 더 많이 추가되어 모델의 복잡도가 커지고 훈련 세트에서의 실수를 바로잡을 기회가 더 많아집니다.

from sklearn.ensemble import GradientBoostingClassifier

# 기본값인 깊이가 3인 트리 100개와 학습률 0.1을 사용
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("훈련 세트 GradientBoosting 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 GradientBoosting 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrtMaxDepth1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrtMaxDepth1.fit(X_train, y_train)
print("훈련 세트 GradientBoosting max_depth=1 정확도: {:.3f}".format(gbrtMaxDepth1.score(X_train, y_train)))
print("테스트 세트 GradientBoosting max_depth=1 정확도: {:.3f}".format(gbrtMaxDepth1.score(X_test, y_test)))

gbrtLearningRate001 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrtLearningRate001.fit(X_train, y_train)
print("훈련 세트 GradientBoosting learning_rate=0.01 정확도: {:.3f}".format(gbrtLearningRate001.score(X_train, y_train)))
print("훈련 세트 GradientBoosting learning_rate=0.01 정확도: {:.3f}".format(gbrtLearningRate001.score(X_test, y_test)))

images.image.plot_feature_importances(gbrt, X_dafaframe, X_dafaframe.columns)
plt.title('heart 데이터로 만든 그래디언트 부스팅 분류기의 특성 중요도')
images.image.save_fig("19.heart_GradientBoosting_feature_importances")  
plt.show()

images.image.plot_feature_importances(gbrtMaxDepth1, X_dafaframe, X_dafaframe.columns)
plt.title('heart 데이터로 만든 그래디언트 부스팅 분류기의 특성 중요도')
images.image.save_fig("19.heart_GradientBoosting_MaxDepth1_feature_importances")  
plt.show()

images.image.plot_feature_importances(gbrtLearningRate001, X_dafaframe, X_dafaframe.columns)
plt.title('heart 데이터로 만든 그래디언트 부스팅 분류기의 특성 중요도')
images.image.save_fig("19.heart_GradientBoosting_LearningRate001_feature_importances")  
plt.show()

# 그래디언트 부스팅 트리 모델의 중요 매개변수
# 1. 트리의 개수를 지정하는 n_estimators
# 2. 이전 트리의 오차를 보정하는 정도를 조절하는 learning_rate




