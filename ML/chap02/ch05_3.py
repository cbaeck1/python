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

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
      stratify=cancer.target, random_state=0)
print("X_train 크기: {} {}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {} {}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {} {}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {} {}".format(y_test.shape, y_test.dtype))

########################################################################
# 5. 결정트리 앙상블 : 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(forest.score(X_test, y_test)))

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
images.image.plot_feature_importances(forest, cancer.data, cancer.feature_names)
plt.title('유방암 데이터로 만든 랜덤 포레스트 모델의 특성 중요도')
images.image.save_fig("3.cancer_random_forest_feature_importances")  
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

images.image.plot_feature_importances(gbrt, cancer.data, cancer.feature_names)
plt.title('유방암 데이터로 만든 그래디언트 부스팅 분류기의 특성 중요도')
images.image.save_fig("3.cancer_GradientBoosting_feature_importances")  
plt.show()

images.image.plot_feature_importances(gbrtMaxDepth1, cancer.data, cancer.feature_names)
plt.title('유방암 데이터로 만든 그래디언트 부스팅 분류기의 특성 중요도')
images.image.save_fig("3.cancer_GradientBoosting_MaxDepth1_feature_importances")  
plt.show()

images.image.plot_feature_importances(gbrtLearningRate001, cancer.data, cancer.feature_names)
plt.title('유방암 데이터로 만든 그래디언트 부스팅 분류기의 특성 중요도')
images.image.save_fig("3.cancer_GradientBoosting_LearningRate001_feature_importances")  
plt.show()

# 그래디언트 부스팅 트리 모델의 중요 매개변수
# 1. 트리의 개수를 지정하는 n_estimators
# 2. 이전 트리의 오차를 보정하는 정도를 조절하는 learning_rate




