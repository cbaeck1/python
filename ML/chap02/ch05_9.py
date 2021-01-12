import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 9. 두 개의 클래스를 가진 2차원 데이터셋 make_moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

########################################################################
# 5. 결정트리 앙상블 : 랜덤 포레스트
# 결정 트리의 주요 단점은 훈련 데이터에 과대적합되는 경향
# 랜덤 포레스트는 이 문제를 회피할 수 있는 방법
# 서로 다른 방향으로 과대적합된 트리를 많이 만들어 평균하여 과대적합을 줄이는 방법

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 각 트리가 고유하게 만들어지도록 무작위한 선택
# 데이터의 부트스트랩 샘플bootstrap sample을 생성
# 무작위로 데이터를 n_samples 횟수만큼 반복 추출, 중복 허용
# 특성을 고르는 것은 매 노드마다 반복되므로 트리의 각 노드는 다른 특성들을 사용
# 같은 결과를 만들어야 한다면 random_state 값을 고정
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

print("훈련 세트 forest 정확도: {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 forest 정확도: {:.3f}".format(forest.score(X_test, y_test)))

# max_features 값을 크게 하면 랜덤 포레스트의 트리들은 매우 비슷해지고 가장 두드러진 특성을 이용해 데이터에 잘 맞추고
# max_features 값을 낮추면 랜덤 포레스트 트리들은 많이 달라지고 각 트리는 데이터에 맞추기 위해 깊이가 깊어지게 됩니다.

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# print(type(axes), type(axes.ravel()))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    # print(type(ax))
    ax.set_title("트리 {}".format(i))
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

# 다섯 개의 랜덤한 결정 트리의 결정 경계와 예측한 확률을 평균내어 만든 결정 경계
mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("랜덤 포레스트")
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

images.image.save_fig("9.5 make_moons_tree_random_forest")  
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






