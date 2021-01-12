import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 6. 세 개의 클래스를 가진 간단한 blobs 데이터셋 centers=4, random_state=8
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, random_state=8)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

########################################################################
# 6. 커널 서포트 벡터 머신 
# 커널 서포트 벡터 머신(보통 그냥 SVM으로 부릅니다)은 입력 데이터에서 단순한 초평면hyperplane으로 정의되지 않는
# 더 복잡한 모델을 만들 수 있도록 확장한 것입니다. 
# 선형 모델을 유연하게 만드는 한 가지 방법은 특성끼리 곱하거나 특성을 거듭제곱하는 식으로 새로운 특성을 추가하는 것

y = y % 2
print(y)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1"])
plt.title('선형적으로 구분되지 않는 클래스를 가진 이진 분류 데이터셋')
images.image.save_fig("6.blobs_2class_Scatter")  
plt.show()

from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1"])
plt.title('선형 SVM으로 만들어진 결정 경계')
images.image.save_fig("6.blobs_linear_svm_Scatter")  
plt.show()


# 두 번째 특성을 제곱하여 추가합니다.
X_new = np.hstack([X, X[:, 1:] ** 2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# 3차원 그래프
ax = Axes3D(figure, elev=-152, azim=-26)
# y == 0인 포인트를 먼저 그리고 그다음 y == 1인 포인트를 그립니다.
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("특성0")
ax.set_ylabel("특성1")
ax.set_zlabel("특성1 ** 2")
plt.title('특성 1에서 유용한 세 번째 특성을 추가하여 [그림 2-37]에서 확장한 데이터셋')
images.image.save_fig("6.blobs_3d_Scatter")  
plt.show()

# 
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# 선형 결정 경계 그리기
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')

ax.set_xlabel("특성0")
ax.set_ylabel("특성1")
ax.set_zlabel("특성1 ** 2")
plt.title('확장된 3차원 데이터셋에서 선형 SVM이 만든 결정 경계')
images.image.save_fig("6.blobs_linear_svm_3d_Scatter")  
plt.show()

# 
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.title('원래 두 개 특성에 투영한 결정 경계')
images.image.save_fig("6.blobs_linear_svm_3d_decision_function_Scatter")  
plt.show()

# 커널 기법 : 새로운 특성을 많이 만들지 않고서도 고차원에서 분류기를 학습
# 실제로 데이터를 확장하지 않고 확장된 특성에 대한 데이터 포인트들의 거리(더 정확히는 스칼라 곱)를 계산
#   1) 다항식 커널 : 원래 특성의 가능한 조합을 지정된 차수까지 모두 계산
#   2) 가우시안 Gaussian 커널 (RBF : radial basis function 커널): 차원이 무한한 특성 공간에 매핑하는 것

# 서포트 벡터 support vector : 두 클래스 사이의 경계에 위치한 데이터 포인트
# 새로운 데이터 포인트에 대해 예측하려면 각 서포트 벡터와의 거리를 측정
# 분류 결정은 서포트 벡터까지의 거리에 기반하며 서포트 벡터의 중요도는 훈련 과정에서 학습하여 
# SVC 객체의 dual_coef_ 속성에 저장한다
# 데이터 포인트 사이의 거리는 가우시안 커널에 의해 계산






