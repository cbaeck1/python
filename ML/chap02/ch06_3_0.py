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

# 3. 위스콘신 유방암 Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer 라고 하겠습니다). 
# 각 종양은 양성 benign (해롭지 않은 종양)과 악성 malignant (암 종양)으로 레이블되어 있고, 
# 조직 데이터를 기반으로 종양이 악성인지를 예측할 수 있도록 학습하는 것이 과제
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))
print(cancer.data, cancer.target)
print(cancer.data[:,:2])

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
    stratify=cancer.target, random_state=0)
print("X_train 크기: {} {}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {} {}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {} {}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {} {}".format(y_test.shape, y_test.dtype))

########################################################################
# 6. 커널 서포트 벡터 머신 
# RBF 커널 SVM을 유방암 데이터셋에 적용해보겠습니다. 기본값 C=1, gamma=1/n_features 를 사용
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

from sklearn.svm import SVC
svc = SVC()
# CValues = np.logspace(-1, 5, 7)
CValues = np.logspace(-4, 5, 200)
gammaValues = np.logspace(-1, 5, 7)
print(CValues, gammaValues)


for gammaValue in gammaValues:
    train_scores = []
    test_scores = []
    for Cvalue in CValues:
        model = SVC(C=Cvalue, gamma=gammaValue)
        train_score = -mean_squared_error(y_train, model.fit(X_train, y_train).predict(X_train))
        test_score = np.mean(cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error", cv=5))
        train_scores.append(train_score)
        test_scores.append(test_score)
    optimal_C = CValues[np.argmax(test_scores)]
    #optimal_gamma = gammaValues[np.argmax(test_scores)]
    optimal_score = np.max(test_scores)

    plt.plot(CValues, test_scores, "-", label="test")
    plt.plot(CValues, train_scores, "*", label="train")
    plt.axhline(optimal_score, linestyle=':')
    plt.axvline(optimal_C, linestyle=':')
    plt.scatter(optimal_C, optimal_score)
    plt.title("Best Regularization")
    plt.ylabel('score')
    plt.xlabel('Regularization')
    plt.legend()
    plt.title("C 값을 가진 SVC 회귀의 계수_gamma_{} 크기 비교".format(gammaValue))
    images.image.save_fig("6.3.breast_cancer_SVC_gamma_{}_coef_compare".format(gammaValue))
    plt.show()

    # parameters = {'C': CValues, 'gamma': gammaValues }
    parameters = {'C': CValues}
    svc_reg = GridSearchCV(svc, parameters, scoring='neg_mean_squared_error',cv=5)
    lr = svc_reg.fit(X_train, y_train)
    print('SVC (gamma={}):'.format(gammaValue), svc_reg.best_params_)
    print('SVC (gamma={}):'.format(gammaValue), svc_reg.best_score_)

# 과대 적합됬음을 알 수 있다.

# SVM을 위한 데이터 전처리
# 커널 SVM에서는 모든 특성 값을 0과 1 사이로 맞추는 방법을 많이 사용

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면 각 특성에 대해 최솟값은 0, 최대값은 1입니다.
X_train_scaled = (X_train - min_on_training) / range_on_training
print("특성별 최소 값\n{}".format(X_train_scaled.min(axis=0)))
print("특성별 최대 값\n {}".format(X_train_scaled.max(axis=0)))

# 테스트 세트에도 같은 작업을 적용 : 훈련 세트에서 계산한 최솟값과 범위를 사용
X_test_scaled = (X_test - min_on_training) / range_on_training
svc_scaled = SVC()

for gammaValue in gammaValues:
    train_scaled_scores = []
    test_scaled_scores = []
    for C in CValues:
        model = SVC(C=C, gamma=gammaValue)
        train_scaled_score = -mean_squared_error(y_train, model.fit(X_train_scaled, y_train).predict(X_train_scaled))
        test_scaled_score = np.mean(cross_val_score(model, X_test_scaled, y_test, scoring="neg_mean_squared_error", cv=5))
        train_scaled_scores.append(train_scaled_score)
        test_scaled_scores.append(test_scaled_score)
    optimal_scaled_C = CValues[np.argmax(test_scaled_scores)]
    # optimal_scaled_gamma = gammaValues[np.argmax(test_scaled_scores)]
    optimal_scaled_score = np.max(test_scaled_scores)

    plt.plot(CValues, test_scaled_scores, "-", label="test_scaled")
    plt.plot(CValues, train_scaled_scores, "*", label="train_scaled")
    plt.axhline(optimal_scaled_score, linestyle=':')
    plt.axvline(optimal_scaled_C, linestyle=':')
    plt.scatter(optimal_scaled_C, optimal_scaled_score)
    plt.title("Best Regularization")
    plt.ylabel('score_scaled')
    plt.xlabel('Regularization')
    plt.legend()
    plt.title("C 값을 가진 SVC_scaled 회귀의 계수_gamma_{} 크기 비교".format(gammaValue))
    images.image.save_fig("6.3.breast_cancer_SVC_scaled_gamma_{}_coef_compare".format(gammaValue))    
    plt.show()

    #parameters = {'C': CValues, 'gamma': gammaValues }
    parameters = {'C': CValues}
    svc_scaled_reg = GridSearchCV(svc, parameters, scoring='neg_mean_squared_error',cv=5)
    lr = svc_scaled_reg.fit(X_train_scaled, y_train)
    print('SVC_scaled:', svc_scaled_reg.best_params_)
    print('SVC_scaled:', svc_scaled_reg.best_score_)


# < 장단점 >
# 데이터의 특성이 몇개 안되더라도 복잡한 결정 경계를 만들 수 있다.
# 저/고 차원이 데이터에서 모두 잘 작동하지만, 샘플이 많은 경우는 잘 맞지않는다.
# 데이터 전처리와 매개변수 석정에 주의해야한다. => 랜덤포레스트나 그래디언트 부스팅을 사용하는 이유

# < 매개변수 >
# 규제 매개변수 값인 C값이 클수록 모델 복잡도는 올라간다.
# RBF커널은 가우시안 커널 폭의 역수인 gamma 매개변수를 더 가진다.
# (SVM에는 RBF커널 말고도 다른 컬널이 많다.)

