import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# 1. 붓꽃iris 데이터셋
# iris_dataset의 키: dict_keys(['target_names', 'feature_names', 'DESCR', 'data', 'target'])
print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 크기: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))

print("target의 타입: {}".format(type(iris_dataset['target'])))
print("target의 크기: {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))

# 성과 측정: 훈련 데이터와 테스트 데이터
# train_test_split 함수의 반환값은 X_train, X_test, y_train, y_test이며 모두 NumPy 배열
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_train 타입: {}".format(type(X_train)))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

########################################################################
# 5. 결정트리 앙상블 : 그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("결정 함수의 결과 형태: {}".format(gbrt.decision_function(X_test).shape))
# decision function 결과 중 앞부분 일부를 확인합니다.
print("결정 함수 결과:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

print("가장 큰 결정 함수의 인덱스:\n{}".format(
    np.argmax(gbrt.decision_function(X_test), axis=1)))
print("예측:\n{}".format(gbrt.predict(X_test)))

# predict_proba 결과 중 앞부분 일부를 확인합니다.
print("예측 확률:\n{}".format(gbrt.predict_proba(X_test)[:6]))
# 행 방향으로 확률을 더하면 1이 됩니다.
print("합: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))

# predict_proba의 결과에 argmax 함수를 적용해서 예측을 재현
print("가장 큰 예측 확률의 인덱스:\n{}".format(
    np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("예측:\n{}".format(gbrt.predict(X_test)))


