import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 19. 클리블랜드(Cleveland) 심장병 재단에서 제공한 작은 데이터셋
# 이 CSV 파일은 수백 개의 행으로 이루어져 있습니다. 
# 각 행은 환자 한 명을 나타내고 각 열은 환자에 대한 속성 값입니다. 
# 이 정보를 사용해 환자의 심장병 발병 여부를 예측
# 1.1 판다스로 데이터프레임 만들기
dataframe = pd.read_csv('data/heart.csv')
print(dataframe.head(), dataframe.shape)

# Categorical
X_dafaframe = dataframe.loc[:,'age':'thal']
X_dafaframe.thal = pd.Categorical(X_dafaframe.thal)
X_dafaframe['thal'] = X_dafaframe.thal.cat.codes
X_dafaframe = X_dafaframe.apply(pd.to_numeric)
print(X_dafaframe.head(), X_dafaframe.shape, X_dafaframe.info())

# numpy array
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X_dafaframe)
X_heart = imputer.transform(X_dafaframe)

Y_heart = dataframe['target']
print(Y_heart.head(), Y_heart.shape)

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, stratify=Y_heart, random_state=66)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))

########################################################################
# 2. 선형분류모델 : 로지스틱, 서포트 벡터 머신 
# C=100 을 사용하니 훈련 세트의 정확도가 높아졌고 테스트 세트의 정확도도 조금 증가
# 복잡도가 높은 모델일수록 성능이 좋음
from sklearn.linear_model import LogisticRegression

plt.figure(figsize=(14, 8))
for C, marker in zip([0.001, 0.01, 0.1, 1, 10, 100], ['.', 'o', '*', '^', 'x', 'v']):
    logreg = LogisticRegression(C=C).fit(X_train, y_train)
    print("2. 선형모델 : C={:.3f} 인 로지스틱 회귀의 훈련/테스트 정확도: {:.3f}/{:.3f}".
        format(C, logreg.score(X_train, y_train), logreg.score(X_test, y_test)))
    plt.plot(logreg.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(dataframe.shape[1]), dataframe.columns, rotation=90)
plt.hlines(0, 0, dataframe.shape[1])
plt.ylim(-2, 2)
plt.xlabel("특성")
plt.ylabel("계수 크기")
plt.legend()
plt.title('heart 데이터셋에 각기 다른 C 값을 사용하여 만든 로지스틱 회귀의 계수')
images.image.save_fig("2.19.heart_logistic_C")  
plt.show()

# LogisticRegression은 기본으로 L2 규제를 적용 : Ridge로 만든 모습과 비슷
# 세 번째 계수(mean perimeter) 
#   C=100, C=1 일 때 이 계수는 음수지만, C=0.001 일 때는 양수가 되며 C=1 일 때보다도 절댓값이 더 큽니다
# texture error특성은 악성인 샘플과 관련이 깊습니다

# L2 규제(몇 개의 특성만 사용)를 사용할 때의 분류 정확도와 계수 그래프 
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
plt.figure(figsize=(14, 8))
for C, marker in zip([0.001, 0.01, 0.1, 1, 10, 100], ['.', 'o', '*', '^', 'x', 'v']):
    lr_l2 = LogisticRegression(C=C, penalty="l2").fit(X_train, y_train)
    lr_none = LogisticRegression(C=C, penalty="none").fit(X_train, y_train)
    print("C={:.3f} 인 L2 로지스틱 회귀의 훈련/테스트 정확도: {:.3f}/{:.3f}".
        format(C, lr_l2.score(X_train, y_train), lr_l2.score(X_test, y_test)))
    print("C={:.3f} 인 penalty=none 로지스틱 회귀의 훈련/테스트 정확도: {:.3f}/{:.3f}".
        format(C, lr_none.score(X_train, y_train), lr_none.score(X_test, y_test)))
    plt.plot(lr_l2.coef_.T, marker, label="C={:.3f}".format(C))
    plt.plot(lr_none.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(dataframe.shape[1]), dataframe.columns, rotation=90)
plt.hlines(0, 0,dataframe.shape[1])
plt.xlabel("특성")
plt.ylabel("계수 크기")
plt.ylim(-2, 2)
plt.legend(loc=3)
plt.title('heart 데이터와 L2,none 규제를 사용하여 각기 다른 C 값을 적용한 로지스틱 회귀 모델의 계수')
images.image.save_fig("2.19.heart_logistic_C_L2")  
plt.show()

# 모델들의 주요 차이는 규제에서 모든 특성을 이용할지 일부 특성만을 사용할지 결정하는 
# penalty 매개변수의 갯수이다


# 다중 클래스 분류용 선형 모델
# 로지스틱 회귀만 제외하고 많은 선형 분류 모델은 태생적으로 이진 분류만을 지원합니다. 다중 클래스를 지원하지 않습니다
# 이진 분류 알고리즘을 다중 클래스 분류 알고리즘으로 확장하는 보편적인 기법은 일대다방법
# 일대다 방식은 각 클래스를 다른 모든 클래스와 구분하도록 이진 분류 모델을 학습시킵니다. 
# 결국 클래스의 수만큼 이진 분류 모델이 만들어집니다. 
# 예측을 할 때 이렇게 만들어진 모든 이진 분류기가 작동하여 가장 높은 점수를 내는 분류기의 클래스를 예측값으로 선택


