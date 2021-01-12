import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 10. 두 개의 클래스를 가진 2차원 데이터셋
X, y = mglearn.tools.make_handcrafted_dataset()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X, y)

########################################################################
# 6. 커널 서포트 벡터 머신 
# 커널 서포트 벡터 머신(보통 그냥 SVM으로 부릅니다)은 입력 데이터에서 단순한 초평면hyperplane으로 정의되지 않는
# 더 복잡한 모델을 만들 수 있도록 확장한 것입니다. 
# 선형 모델을 유연하게 만드는 한 가지 방법은 특성끼리 곱하거나 특성을 거듭제곱하는 식으로 새로운 특성을 추가하는 것

# 커널 기법 : 새로운 특성을 많이 만들지 않고서도 고차원에서 분류기를 학습
# 실제로 데이터를 확장하지 않고 확장된 특성에 대한 데이터 포인트들의 거리(더 정확히는 스칼라 곱)를 계산
#   1) 다항식 커널 : 원래 특성의 가능한 조합을 지정된 차수까지 모두 계산
#   2) 가우시안 Gaussian 커널 (RBF : radial basis function 커널): 차원이 무한한 특성 공간에 매핑하는 것

# 서포트 벡터 support vector : 두 클래스 사이의 경계에 위치한 데이터 포인트
# 새로운 데이터 포인트에 대해 예측하려면 각 서포트 벡터와의 거리를 측정
# 분류 결정은 서포트 벡터까지의 거리에 기반하며 서포트 벡터의 중요도는 훈련 과정에서 학습하여 
# SVC 객체의 dual_coef_ 속성에 저장한다
# 데이터 포인트 사이의 거리는 가우시안 커널에 의해 계산
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# 서포트 벡터
sv = svm.support_vectors_
# dual_coef_의 부호에 의해 서포트 벡터의 클래스 레이블이 결정됩니다.
# SVM은 매우 부드럽고 비선형(직선이 아닌) 경계를 생성
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=2, markeredgewidth=3)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.title('RBF 커널을 사용한 SVM으로 만든 결정 경계와 서포트 벡터')
images.image.save_fig("10.make_handcrafted_rbf_Scatter")  
plt.show()


# SVM 매개변수 튜닝
# gamma 매개변수는 앞 절의 공식에 나와 있는 γ로 가우시안 커널 폭의 역수
#  -> 하나의 훈련 샘플이 미치는 영향의 범위를 결정
# C 매개변수는 선형 모델에서 사용한 것과 비슷한 규제 매개변수
#  -> 각 포인트의 중요도(정확히는 dual_coef_ 값)를 제한

# 왼쪽에서 오른쪽으로 가면서 gamma 매개변수를 0.1에서 10으로 증가
# 왼쪽 그림의 결정 경계는 매우 부드럽고 오른쪽으로 갈수록 결정 경계는 하나의 포인트에 더 민감
# 위에서 아래로는 C 매개변수를 0.1에서 1000으로 증가
# 왼쪽 위의 결정 경계는 거의 선형에 가까우며 잘못 분류된 데이터 포인트가 경계에 거의 영향을 주지 않습니다. 
# 왼쪽 아래에서 볼 수 있듯이 C를 증가시키면 이 포인트들이 모델에 큰 영향을 주며 결정 경계를 휘어서 정확하게 분류
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["클래스 0", "클래스 1", "클래스 0 서포트 벡터", "클래스 1 서포트 벡터"], ncol=4, loc=(.9, 1.2))
plt.title('C와 gamma 매개변수 설정에 따른 결정 경계와 서포트 벡터')
images.image.save_fig("10.make_handcrafted_gamma_C_Scatter")  
plt.show()

# SVM을 위한 데이터 전처리
# 커널 SVM에서는 모든 특성 값을 0과 1 사이로 맞추는 방법을 많이 사용

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면
# 각 특성에 대해 최솟값은 0, 최대값은 1입니다.
X_scaled = (X - min_on_training) / range_on_training
print("특성별 최소 값\n{}".format(X_scaled.min(axis=0)))
print("특성별 최대 값\n {}".format(X_scaled.max(axis=0)))

svc = SVC()
svc.fit(X_scaled, y)
print("훈련 세트 정확도: {:.3f}".format(svc.score(X_scaled, y)))

svc = SVC(C=1000)
svc.fit(X_scaled, y)
print("훈련 세트 정확도: {:.3f}".format(svc.score(X_scaled, y)))
