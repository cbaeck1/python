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

# 20. life

# 1.1 판다스로 데이터프레임 만들기
dataframe = pd.read_csv('d:/data/trans서울중구.csv')
print(dataframe.head(), dataframe.shape)

# Categorical
X_dataframe = dataframe.loc[:,'매출_금액':'industry']
X_dataframe.industry = pd.Categorical(X_dataframe.industry)
X_dataframe.industry = X_dataframe.industry.cat.codes
X_dataframe = X_dataframe.apply(pd.to_numeric)
print(X_dataframe.head(), X_dataframe.shape, X_dataframe.info())

# numpy array
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X_dataframe)
X_life = imputer.transform(X_dataframe)

Y_life = dataframe['TARGET']
print(Y_life.head(), Y_life.shape)

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_life, Y_life, stratify=Y_life, random_state=66)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))

########################################################################
# 6. 커널 서포트 벡터 머신 
# RBF 커널 SVM을 life 데이터셋에 적용해보겠습니다. 기본값 C=1, gamma=1/n_features 를 사용
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print("훈련/테스트 세트 정확도 {:.5f}/{:.5f}/{:.5f}".format(svc.score(X_train, y_train), svc.score(X_test, y_test), svc.n_features_in_))

train_scores = []
test_scores = []
CValues = np.logspace(-25, 1, 400)
print('C:', CValues[0],  CValues[-1])

plt.figure(figsize=(14, 8))
i = 0
for C in CValues:
    svc = SVC(C=C)
    history = svc.fit(X_train, y_train)
    train_score = svc.score(X_train, y_train)
    train_scores.append(train_score)
    test_score = svc.score(X_test, y_test)
    test_scores.append(test_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.25f} gamma={:.25f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(C, svc._gamma, train_score, test_score))
    i = i + 1

optimal_C = CValues[np.argmax(test_scores)]
svc = SVC(C=optimal_C)
history = svc.fit(X_train, y_train)
print("6.선형분류모델 : optimal C={:.25f} gamma={:.25f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, svc._gamma, svc.score(X_train, y_train), svc.score(X_test, y_test)))
plt.plot(CValues, train_scores, '-', label="훈련 정확도")
plt.plot(CValues, test_scores, '--', label="테스트 정확도")
plt.axvline(optimal_C, linestyle=':')
plt.xlabel("C")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 다른 C 값 gamma={:.25f}을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(svc._gamma))
images.image.save_fig("6.20.life_SVC_C_score")  
plt.show()

# 
gammaValues = np.linspace(svc._gamma/100, svc._gamma*100000000000, 400)
print('gamma:', gammaValues[0], gammaValues[-1])

train_scores = []
test_scores = []
plt.figure(figsize=(14, 8))
i = 0
for gamma in gammaValues:
    svc = SVC(C=optimal_C, gamma=gamma)
    history = svc.fit(X_train, y_train)
    train_score = svc.score(X_train, y_train)
    train_scores.append(train_score)
    test_score = svc.score(X_test, y_test)
    test_scores.append(test_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.25f} gamma={:.25f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(optimal_C, gamma, train_score, test_score))
    i = i + 1

optimal_gamma = gammaValues[np.argmax(test_scores)]
svc = SVC(C=optimal_C, gamma=optimal_gamma)
history = svc.fit(X_train, y_train)
print("6.선형분류모델 : optimal C={:.25f} gamma={:.25f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, optimal_gamma, svc.score(X_train, y_train), svc.score(X_test, y_test)))
plt.plot(gammaValues, train_scores, '-', label="훈련 정확도")
plt.plot(gammaValues, test_scores, '--', label="테스트 정확도")
plt.axvline(optimal_gamma, linestyle=':')
plt.xlabel("gamma")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 C={:.25f} 값 다른 gamma 을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(optimal_C))
images.image.save_fig("6.20.life_SVC_gamma_score")  
plt.show()


# 과대 적합됬음을 알 수 있다.
# SVM은 매개변수와 데이터 스케일에 매우 민감하다
# 각 특성의 최대.최소값을 로그스케일로 표현

plt.boxplot(X_train, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("특성 목록")
plt.ylabel("특성 크기")
plt.title('life 데이터셋의 특성 값 범위(y 축은 로그 스케일)')
images.image.save_fig("6.20.life_rbf_Scatter")  
plt.show()

# SVM을 위한 데이터 전처리
# 커널 SVM에서는 모든 특성 값을 0과 1 사이로 맞추는 방법을 많이 사용

# 훈련 세트에서 특성별 최솟값 계산
min_on_training = X_train.min(axis=0)
# 훈련 세트에서 특성별 (최댓값 - 최솟값) 범위 계산
range_on_training = (X_train - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면
# 각 특성에 대해 최솟값은 0, 최대값은 1입니다.
X_train_scaled = (X_train - min_on_training) / range_on_training
print("특성별 최소 값\n{}".format(X_train_scaled.min(axis=0)))
print("특성별 최대 값\n {}".format(X_train_scaled.max(axis=0)))

# 테스트 세트에도 같은 작업을 적용 : 훈련 세트에서 계산한 최솟값과 범위를 사용
X_test_scaled = (X_test - min_on_training) / range_on_training

#  
svc_scaled = SVC()
svc_scaled.fit(X_train_scaled, y_train)
print("훈련/테스트 세트 정확도 {:.5f}/{:.5f}".format(svc_scaled.score(X_train_scaled, y_train), svc_scaled.score(X_test_scaled, y_test)))

train_scaled_scores = []
test_scaled_scores = []
CValues = np.logspace(-4, 4, 400)
print('C:', CValues[0],  CValues[-1])

plt.figure(figsize=(14, 8))
i = 0
for C in CValues:
    svc_scaled = SVC(C=C)
    history = svc_scaled.fit(X_train_scaled, y_train)
    train_scaled_score = svc_scaled.score(X_train_scaled, y_train)
    train_scaled_scores.append(train_scaled_score)
    test_scaled_score = svc_scaled.score(X_test_scaled, y_test)
    test_scaled_scores.append(test_scaled_score)
    if i%20 == 0:
        print("6.선형분류모델 : C={:.25f} gamma={:.25f}인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(C, svc_scaled._gamma, train_scaled_score, test_scaled_score))
    i = i + 1

optimal_C = CValues[np.argmax(test_scaled_scores)]
svc_scaled = SVC(C=optimal_C)
history = svc_scaled.fit(X_train_scaled, y_train)
print("6.선형분류모델 : optimal C={:.25f} gamma={:.25f}인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, svc_scaled._gamma, svc_scaled.score(X_train_scaled, y_train), svc_scaled.score(X_test_scaled, y_test)))
plt.plot(CValues, train_scaled_scores, '-', label="훈련 정확도")
plt.plot(CValues, test_scaled_scores, '--', label="테스트 정확도")
plt.axvline(optimal_C, linestyle=':')
plt.xlabel("C")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋(최댓값 - 최솟값)에 각기 다른 C 값을 사용하여 만든 SVC의 훈련/테스트 정확도')
images.image.save_fig("6.20.life_scaled_SVC_C_score")  
plt.show()

gammaValues = np.linspace(svc_scaled._gamma/100000, svc_scaled._gamma*10, 400)
print('gamma:', gammaValues[0], gammaValues[-1])

train_scaled_scores = []
test_scaled_scores = []
plt.figure(figsize=(14, 8))
i = 0
for gamma in gammaValues:
    svc_scaled = SVC(C=optimal_C, gamma=gamma)
    history = svc_scaled.fit(X_train_scaled, y_train)
    train_scaled_score = svc_scaled.score(X_train_scaled, y_train)
    train_scaled_scores.append(train_scaled_score)
    test_scaled_score = svc_scaled.score(X_test_scaled, y_test)
    test_scaled_scores.append(test_scaled_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.25f} gamma={:.25f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(optimal_C, gamma, train_scaled_score, test_scaled_score))
    i = i + 1

optimal_gamma = gammaValues[np.argmax(test_scaled_scores)]
svc_scaled = SVC(C=optimal_C, gamma=optimal_gamma)
history = svc_scaled.fit(X_train_scaled, y_train)
print("6.선형분류모델 : optimal C={:.25f} gamma={:.25f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, optimal_gamma, svc_scaled.score(X_train_scaled, y_train), svc_scaled.score(X_test_scaled, y_test)))
plt.plot(gammaValues, train_scaled_scores, '-', label="훈련 정확도")
plt.plot(gammaValues, test_scaled_scores, '--', label="테스트 정확도")
plt.axvline(optimal_gamma, linestyle=':')
plt.xlabel("gamma")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 C={:.25f} 값 다른 gamma 을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(optimal_C))
images.image.save_fig("6.20.life_scaled_SVC_gamma_score")  
plt.show()

# < 장단점 >
# 데이터의 특성이 몇개 안되더라도 복잡한 결정 경계를 만들 수 있다.
# 저/고 차원이 데이터에서 모두 잘 작동하지만, 샘플이 많은 경우는 잘 맞지않는다.
# 데이터 전처리와 매개변수 석정에 주의해야한다. => 랜덤포레스트나 그래디언트 부스팅을 사용하는 이유

# < 매개변수 >
# 규제 매개변수 값인 C값이 클수록 모델 복잡도는 올라간다.
# RBF커널은 가우시안 커널 폭의 역수인 gamma 매개변수를 더 가진다.
# (SVM에는 RBF커널 말고도 다른 컬널이 많다.)
