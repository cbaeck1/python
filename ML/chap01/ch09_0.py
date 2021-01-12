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
print(X[:5], y[:5])

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.title("결정 트리를 적용할 반달 모양의 데이터셋")
images.image.save_fig("9. make_moons_Scatter")  
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("X_train 타입: {}".format(type(X_train)))
print("y_train 타입: {}".format(type(y_train)))
print(X_train[:5], y_train[:5])

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.title("결정 트리를 적용할 반달 모양의 데이터셋(X_train)")
images.image.save_fig("9. make_moons_Scatter_by_X_train")  
plt.show()

# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([X, X_train, X_test], [y, y_train, y_test], ['전체','X_train','X_test'], axes):
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title("{}".format(title))
  ax.set_xlabel("특성 0")
  ax.set_ylabel("특성 1")

axes[0].legend(loc=3)
images.image.save_fig("9. make_moons_scatter_compare")  
plt.show()

