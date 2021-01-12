import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image


# 2. 두 개의 특성을 가진 forge 데이터셋
X, y = mglearn.datasets.make_forge()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X, y)
print(X[:, 0], X[:, 1])

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.title("Forge Scatter Plot")
images.image.save_fig("2.Forge_Scatter")  
plt.show()

########################################################################
# 2. 선형분류모델 : 로지스틱, 서포트 벡터 머신 
# 예측한 값을 임계치 0과 비교 0보다 작으면 클래스를 -1이라고 예측하고 0보다 크면 +1이라고 예측
# 분류용 선형 모델에서는 결정 경계가 입력의 선형 함수
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# forge 데이터셋에 기본 매개변수를 사용해 만든 선형 SVM과 로지스틱 회귀 모델의 결정 경계
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend()
images.image.save_fig("2.2.Forge_svc_logistic")  
plt.show()

# LinearSVC 와 LogisticRegression으로 만든 결정 경계가 직선으로 표현
# LogitsticRegression과 LinearSVC에서 규제의 강도를 결정하는 매개변수는 C입니다.
# C의 값이 높아지면 규제가 감소
# 알고리즘은 C의 값이 낮아지면 데이터 포인트 중 다수에 맞추려고 하는 반면, 
#           C의 값을 높이면 개개의 데이터 포인트를 정확히 분류하려고 노력
# LinearSVC를 사용한 예
# forge 데이터셋에 각기 다른 C 값으로 만든 선형 SVM 모델의 결정 경계
mglearn.plots.plot_linear_svc_regularization()
images.image.save_fig("2.2.Forge_svc_C")  
plt.show()

