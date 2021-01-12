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

# 8. 메모리 가격 동향 데이터 셋 ram_prices
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
print("데이터의 형태: {}".format(ram_prices.shape))
print("ram_prices 타입: {}".format(type(ram_prices)))
print(ram_prices)

########################################################################
# 4. 결정트리 vs 선형모델
# 날짜 특성 하나만으로 2000년 전까지의 데이터로부터 2000년 후의 가격을 예측
from sklearn.tree import DecisionTreeRegressor
# 2000년 이전을 훈련 데이터로, 2000년 이후를 테스트 데이터로 만듭니다.
data_train = ram_prices[ram_prices.date < 2000] 
data_test = ram_prices[ram_prices.date >= 2000]
# 가격 예측을 위해 날짜 특성만을 이용합니다.
X_train = data_train.date[:, np.newaxis]
# 데이터와 타깃의 관계를 간단하게 만들기 위해 로그 스케일로 바꿉니다.
y_train = np.log(data_train.price)
#
X_test = data_test.date[:, np.newaxis]
y_test = np.log(data_test.price)

print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("X_train 타입: {}".format(type(X_train)))
print("y_train 타입: {}".format(type(y_train)))
print(X_train, y_train, X_test, y_test)

tree = DecisionTreeRegressor().fit(X_train, y_train)
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression().fit(X_train, y_train)

# 예측은 전체 기간에 대해서 수행합니다.
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# 예측한 값의 로그 스케일을 되돌립니다.
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# 트리 모델은 훈련 데이터 밖의 새로운 데이터를 예측할 능력이 없습니다
plt.semilogy(data_train.date, data_train.price, label="훈련 데이터")
plt.semilogy(data_test.date, data_test.price, label="테스트 데이터")
plt.semilogy(ram_prices.date, price_tree, label="트리 예측")
plt.semilogy(ram_prices.date, price_lr, label="선형 회귀 예측")
plt.legend()
plt.title("램 가격 데이터를 사용해 만든 선형 모델과 회귀 트리의 예측값 비교")
images.image.save_fig("8.ram_prices_predict_compare")  
plt.show()

# 장단점과 매개변수
#   1. 모델을 쉽게 시각화할 수 있어서 비전문가도 이해하기 쉽습니다(비교적 작은 트리일 때).
#   2. 데이터의 스케일에 구애받지 않습니다. 각 특성이 개별적으로 처리되어 데이터를 분할하는 데 
#   데이터 스케일의 영향을 받지 않으므로 결정 트리에서는 특성의 정규화나 표준화 같은 전처리 과정이 필요 없습니다. 
#   3. 특히 특성의 스케일이 서로 다르거나 이진 특성과 연속적인 특성이 혼합되어 있을 때도 잘 작동합니다.
#   
#   1. 결정 트리의 주요 단점은 사전 가지치기를 사용함에도 불구하고 과대적합되는 경향이 있어 일반화 성능이 좋지 않다