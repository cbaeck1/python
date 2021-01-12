import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 11. 두 개의 클래스를 가진 2차원 데이터셋 make_circles
from sklearn.datasets import make_circles
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
    train_test_split(X, y_named, y, random_state=0)
# 예제를 위해 클래스의 이름을 "blue"와 "red"로 바꿉니다.
y_named = np.array(["blue", "red"])[y]

########################################################################
# 5. 결정트리 앙상블 : 그래디언트 부스팅
# 그래디언트 부스팅 모델을 만듭니다.
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)
print("X_test.shape: {}".format(X_test.shape))
# 이진 분류에서 decision_function 반환값의 크기는 (n_samples,)이며 각 샘플이 하나의 실수 값을 반환
print("결정 함수 결과 형태: {}".format(gbrt.decision_function(X_test).shape))
# 결정 함수 결과 중 앞부분 일부를 확인합니다.
print("결정 함수:\n{}".format(gbrt.decision_function(X_test)[:6]))
# 결정 함수의 부호만 보고 예측 결과를 알 수 있습니다.
print("임계치와 결정 함수 결과 비교:\n{}".format(gbrt.decision_function(X_test) > 0))
print("예측:\n{}".format(gbrt.predict(X_test)))

# 이진 분류 불확실성 추정
# 이진 분류에서 음성 클래스는 항상 classes_ 속성의 첫 번째 원소이고 양성 클래스는 classes_의 두 번째 원소입니다. 
# 그래서 predict 함수의 결과를 완전히 재현하려면 classes_ 속성을 사용하면 됩니다.
# 불리언 값을 0과 1로 변환합니다.
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
# classes_에 인덱스로 사용합니다.
pred = gbrt.classes_[greater_zero]
# pred와 gbrt.predict의 결과를 비교합니다.
print("pred는 예측 결과와 같다: {}".format( np.all(pred == gbrt.predict(X_test))))

decision_function = gbrt.decision_function(X_test)
print("결정 함수 최솟값: {:.2f} 최댓값: {:.2f}".format(
    np.min(decision_function), np.max(decision_function)))

# 2차원 평면의 모든 점에 대해 decision_function의 값을 색으로 표현
# 훈련 데이터는 원 모양이고 테스트 데이터는 삼각형
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)
for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["테스트 클래스 0", "테스트 클래스 1", "훈련 클래스 0",
                "훈련 클래스 1"], ncol=4, loc=(.1, 1.1))
plt.title('2차원 예제 데이터셋을 사용해 만든 그래디언트 부스팅 모델의 결정 경계(좌)와 결정 함수(우)')
images.image.save_fig("11.make_circles_decision_function_compare")  
plt.show()

# 예측 확률 : predict_proba의 출력은 각 클래스에 대한 확률
print("확률 값의 형태: {}".format(gbrt.predict_proba(X_test).shape))
# predict_proba 결과 중 앞부분 일부를 확인합니다.
print("예측 확률:\n{}".format(gbrt.predict_proba(X_test[:6])))

# 결정 경계와 클래스 1의 확률
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(
    gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["테스트 클래스 0", "테스트 클래스 1", "훈련 클래스 0",
                "훈련 클래스 1"], ncol=4, loc=(.1, 1.1))
plt.title('그래디언트 부스팅 모델의 결정 경계(좌)와 예측 확률(우)')
images.image.save_fig("11.make_circles_predict_proba_compare")  
plt.show()

# 다중 분류에서의 불확실성
# 






