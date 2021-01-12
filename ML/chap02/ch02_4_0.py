import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격 Boston Housing 데이터셋
# 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 1970년대 보스턴 주변의 주택 평균 가격을 예측
# 이 데이터셋에는 데이터 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['DESCR']+ "\n...")
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.data[:,:2])

# 특성 공학feature engineering : load_extended_boston
# 13개의 원래 특성에 13개에서 2개씩 (중복을 포함해) 짝지은 91개의 특성을 더해 총 104개가 됩니다.
# X, y : numpy ndarray
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {} {}".format(X.shape, X.dtype))
print("y.shape: {} {}".format(y.shape, y.dtype))
#print("특성 이름:\n{}".format(X.column_names))
print(X, y)

# 훈련 세트, 테스트 세트 random_state=0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_train 크기: {} {}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {} {}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {} {}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {} {}".format(y_test.shape, y_test.dtype))
# print(X_train[:, 0], X_train[:, 1], y_train)

########################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

# 2. 선형모델 : 릿지 회귀  
from sklearn.linear_model import Ridge
max_iter = 1000
ridge = Ridge(max_iter=max_iter)
alphas = np.logspace(-4, 0, 200)
print(alphas)

train_scores = []
test_scores = []
for alpha in alphas:
    model = Ridge(alpha=alpha, max_iter=max_iter)
    train_score = -mean_squared_error(y, model.fit(X, y).predict(X))
    train_scores.append(train_score)
    test_score = np.mean(cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    test_scores.append(test_score)
optimal_alpha = alphas[np.argmax(test_scores)]
optimal_score = np.max(test_scores)

plt.plot(alphas, test_scores, "-", label="test")
plt.plot(alphas, train_scores, "--", label="train")
plt.axhline(optimal_score, linestyle=':')
plt.axvline(optimal_alpha, linestyle=':')
plt.scatter(optimal_alpha, optimal_score)
plt.title("Best Regularization")
plt.ylabel('score')
plt.xlabel('Regularization')
plt.legend()
plt.title("alpha 값을 가진 Ridge 회귀의 계수 크기 비교")
images.image.save_fig("2.4.boston_Ridge_alpha_coef_compare")
plt.show()

parameters = {'alpha': alphas }
ridge_reg = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error',cv=5)
lr = ridge_reg.fit(X,y)
print('Ridge:',ridge_reg.best_params_)
print('Ridge:',ridge_reg.best_score_)

# 2. 선형모델 : 라쏘 회귀
from sklearn.linear_model import Lasso

train_scores = []
test_scores = []
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=max_iter)
    train_score = -mean_squared_error(y, model.fit(X, y).predict(X))
    test_score = np.mean(cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    train_scores.append(train_score)
    test_scores.append(test_score)
optimal_alpha = alphas[np.argmax(test_scores)]
optimal_score = np.max(test_scores)

plt.plot(alphas, test_scores, "-", label="test")
plt.plot(alphas, train_scores, "--", label="train")
plt.axhline(optimal_score, linestyle=':')
plt.axvline(optimal_alpha, linestyle=':')
plt.scatter(optimal_alpha, optimal_score)
plt.title("Best Regularization")
plt.ylabel('score')
plt.xlabel('Regularization')
plt.legend()
plt.title("alpha 값을 가진 Lasso 회귀의 계수 크기 비교")
images.image.save_fig("2.4.boston_Lasso_alpha_coef_compare")
plt.show()

lasso = Lasso(max_iter=max_iter)
parameters = {'alpha': alphas} 
lasso_reg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error',cv=5)
lasso_reg.fit(X,y)
print('Lasso:',lasso_reg.best_params_)
print('Lasso:',lasso_reg.best_score_)


########################################################################
# 2. 선형모델 : sklearn.linear_model.ElasticNet
from sklearn.linear_model import ElasticNet

train_scores = []
test_scores = []
for alpha in alphas:
    model = ElasticNet(alpha=alpha, max_iter=max_iter)
    train_score = -mean_squared_error(y, model.fit(X, y).predict(X))
    test_score = np.mean(cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    train_scores.append(train_score)
    test_scores.append(test_score)
optimal_alpha = alphas[np.argmax(test_scores)]
optimal_score = np.max(test_scores)

plt.plot(alphas, test_scores, "-", label="test")
plt.plot(alphas, train_scores, "--", label="train")
plt.axhline(optimal_score, linestyle=':')
plt.axvline(optimal_alpha, linestyle=':')
plt.scatter(optimal_alpha, optimal_score)
plt.title("Best Regularization")
plt.ylabel('score')
plt.xlabel('Regularization')
plt.legend()
plt.title("alpha 값을 가진 ElasticNet 회귀의 계수 크기 비교")
images.image.save_fig("2.4.boston_ElasticNet_alpha_coef_compare")
plt.show()

elasticnet = ElasticNet(max_iter=max_iter)
parameters = {'alpha': alphas} 
elasticnet_reg = GridSearchCV(elasticnet, parameters, scoring='neg_mean_squared_error',cv=5)
elasticnet_reg.fit(X,y)
print('ElasticNet:',elasticnet_reg.best_params_)
print('ElasticNet:',elasticnet_reg.best_score_)

