import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 5. 선형 회귀(최소제곱법)을 위한 wave  데이터셋. n_samples = 40
# 
X, y = mglearn.datasets.make_wave(n_samples=40)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:5], y[:5])

# ŷ = w[0] × x[0] + b
# 1차원 wave 데이터셋으로 파라미터 w[0]와 b를 직선이 되도록 학습
# 1. k-최근접 이웃 알고리즘 : 회귀 다음에서 실행하면 아래 에러가 발생
# Exception has occurred: ValueError illegal value in 4-th argument of internal None 
# On entry to DLASCLS parameter number  4 had an illegal value
mglearn.plots.plot_linear_regression_wave()
plt.title('wave 데이터셋에 대한 선형 모델의 예측')
images.image.save_fig("5.Make_Wave_regression")  
plt.show()

# 산점도 : 2개의 특성
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("특성")
plt.ylabel("타깃")
plt.title("Make Wave Scatter Plot")
images.image.save_fig("5.Make_Wave_Scatter")  
plt.show()

# 1. k-최근접 이웃 알고리즘 : 회귀
mglearn.plots.plot_knn_regression(n_neighbors=1)
images.image.save_fig("5.Make_Wave_knn_regression_n_neighbors_1")  
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
images.image.save_fig("5.Make_Wave_knn_regression_n_neighbors_3")  
plt.show()

# wave 데이터셋을 훈련 세트와 테스트 세트로 나눕니다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))


# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([X, X_train, X_test], [y, y_train, y_test], ['전체','X_train','X_test'], axes):
  # 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
  mglearn.discrete_scatter(X, y, ax=ax)
  ax.set_title("{}".format(title))
  ax.set_xlabel("특성 1")
  ax.set_ylabel("특성 2")

axes[0].legend(loc=3)
images.image.save_fig("5.Make_Wave_Scatter_compare")  
plt.show()


