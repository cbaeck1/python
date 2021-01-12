import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 2. 두 개의 특성을 가진 forge 데이터셋은 인위적으로 만든 이진 분류 데이터셋
X, y = mglearn.datasets.make_forge()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:, 0], X[:, 1], y)

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.title("Forge Scatter Plot")
images.image.save_fig("2.Forge_Scatter")  
plt.show()

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.title("Forge Scatter Plot")
images.image.save_fig("2.Forge_Scatter_by_X_train")  
plt.show()

# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([X, X_train, X_test], [y, y_train, y_test], ['전체','X_train','X_test'], axes):
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title("{}".format(title))
  ax.set_xlabel("특성 0")
  ax.set_ylabel("특성 1")

axes[0].legend(loc=3)
images.image.save_fig("2.Forge_scatter_compare")  
plt.show()



