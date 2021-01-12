import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 20. life

# 1.1 판다스로 데이터프레임 만들기
dataframe = pd.read_csv('e:/data/trans서울중구.csv')
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
rs = 5
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_life, Y_life, stratify=Y_life, random_state=rs)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))


#################################################################################
# 유방암 데이터셋 커널SVM(SVC)를 적용하고 MinmaxScaler 를 이용
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 3. MinMaxScaler 의 fit 메서드는 훈련세트에 있는 특성마다 최대/최소값을 계산한다.
scaler.fit(X_train)
# MinMaxScaler(copy=True, feature_range=(0, 1))
# fit 메서드로 학습한 변환을 적용하려면 스케일 객체의 transform 메서드를 사용한다.
# 데이터 변환
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일이 조정된 후 데이터셋의 속성을 출력
print("변환된 후 크기: {}".format(X_train_scaled.shape))
# 변환된 후 크기: (426, 30)
# 데이터 스케일을 변환해도 개수에는 변화 없다. 
print("스케일 조정 전 특성별 최소값: {}".format(X_train.min(axis=0)))
print("스케일 조정 전 특성별 최대값: {}".format(X_train.max(axis=0)))
print("스케일 조정 후 X_train 특성별 최소값: {}".format(X_train_scaled.min(axis=0)))
print("스케일 조정 후 X_train 특성별 최대값: {}".format(X_train_scaled.max(axis=0)))
print("스케일 조정 후 X_test 특성별 최소값: \n{}".format(X_test_scaled.min(axis=0)))
print("스케일 조정 후 X_test 특성별 최대값: \n{}".format(X_test_scaled.max(axis=0)))

#################################################################################
# 1. StandardScaler 를 사용
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)

X_train_scaled_standard = standard_scaler.transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)

#################################################################################
# 6. 커널 서포트 벡터 머신 
# 데이터 전처리 효과를 성능 측정
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_scaled_standard, y_train)
print("훈련/테스트 세트 정확도 {:.5f}/{:.5f}".format(svc.score(X_train_scaled_standard, y_train), svc.score(X_test_scaled_standard, y_test)))

train_scores = []
test_scores = []
CValues = np.logspace(-1, 1, 400)
print('C:', CValues[0],  CValues[-1])

plt.figure(figsize=(14, 8))
i = 0
for C in CValues:
    svc = SVC(C=C)
    history = svc.fit(X_train_scaled_standard, y_train)
    train_score = svc.score(X_train_scaled_standard, y_train)
    train_scores.append(train_score)
    test_score = svc.score(X_test_scaled_standard, y_test)
    test_scores.append(test_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(C, svc._gamma, train_score, test_score))
    i = i + 1

optimal_C = CValues[np.argmax(test_scores)]
svc = SVC(C=optimal_C)
history = svc.fit(X_train_scaled_standard, y_train)
print("6.선형분류모델 : optimal C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, svc._gamma, svc.score(X_train_scaled_standard, y_train), svc.score(X_test_scaled_standard, y_test)))
plt.plot(CValues, train_scores, '-', label="훈련 정확도")
plt.plot(CValues, test_scores, '--', label="테스트 정확도")
plt.axvline(optimal_C, linestyle=':')
plt.xlabel("C")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 다른 C 값 gamma={:.12f}을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(svc._gamma))
images.image.save_fig("6.20.life_SVC_C_score")  
plt.show()

# 
gammaValues = np.linspace(svc._gamma/10, svc._gamma*1000, 400)
print('gamma:', gammaValues[0], gammaValues[-1])

train_scores = []
test_scores = []
plt.figure(figsize=(14, 8))
i = 0
for gamma in gammaValues:
    svc = SVC(C=optimal_C, gamma=gamma)
    history = svc.fit(X_train_scaled_standard, y_train)
    train_score = svc.score(X_train_scaled_standard, y_train)
    train_scores.append(train_score)
    test_score = svc.score(X_test, y_test)
    test_scores.append(test_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(optimal_C, gamma, train_score, test_score))
    i = i + 1

optimal_gamma = gammaValues[np.argmax(test_scores)]
svc = SVC(C=optimal_C, gamma=optimal_gamma)
history = svc.fit(X_train_scaled_standard, y_train)
print("6.선형분류모델 : optimal C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, optimal_gamma, svc.score(X_train_scaled_standard, y_train), svc.score(X_test_scaled_standard, y_test)))
plt.plot(gammaValues, train_scores, '-', label="훈련 정확도")
plt.plot(gammaValues, test_scores, '--', label="테스트 정확도")
plt.axvline(optimal_gamma, linestyle=':')
plt.xlabel("gamma")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 C={:.12f} 값 다른 gamma 을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(optimal_C))
images.image.save_fig("6.20.life_SVC_gamma_score")  
plt.show()

###################################################################
svc = SVC()
svc.fit(X_train_scaled, y_train)
print("훈련/테스트 세트 정확도 {:.5f}/{:.5f}".format(svc.score(X_train_scaled, y_train), svc.score(X_test_scaled, y_test)))

train_scores = []
test_scores = []
CValues = np.logspace(-1, 1, 400)
print('C:', CValues[0],  CValues[-1])

plt.figure(figsize=(14, 8))
i = 0
for C in CValues:
    svc = SVC(C=C)
    history = svc.fit(X_train_scaled, y_train)
    train_score = svc.score(X_train_scaled, y_train)
    train_scores.append(train_score)
    test_score = svc.score(X_train_scaled, y_test)
    test_scores.append(test_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(C, svc._gamma, train_score, test_score))
    i = i + 1

optimal_C = CValues[np.argmax(test_scores)]
svc = SVC(C=optimal_C)
history = svc.fit(X_train_scaled, y_train)
print("6.선형분류모델 : optimal C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, svc._gamma, svc.score(X_train_scaled, y_train), svc.score(X_test_scaled, y_test)))
plt.plot(CValues, train_scores, '-', label="훈련 정확도")
plt.plot(CValues, test_scores, '--', label="테스트 정확도")
plt.axvline(optimal_C, linestyle=':')
plt.xlabel("C")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 다른 C 값 gamma={:.12f}을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(svc._gamma))
images.image.save_fig("6.20.life_SVC_C_score")  
plt.show()

# 
gammaValues = np.linspace(svc._gamma/10, svc._gamma*1000, 400)
print('gamma:', gammaValues[0], gammaValues[-1])

train_scores = []
test_scores = []
plt.figure(figsize=(14, 8))
i = 0
for gamma in gammaValues:
    svc = SVC(C=optimal_C, gamma=gamma)
    history = svc.fit(X_train_scaled, y_train)
    train_score = svc.score(X_train_scaled, y_train)
    train_scores.append(train_score)
    test_score = svc.score(X_test, y_test)
    test_scores.append(test_score)
    if i%20 == 0:
        print("6.선형분류모델 C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
            format(optimal_C, gamma, train_score, test_score))
    i = i + 1

optimal_gamma = gammaValues[np.argmax(test_scores)]
svc = SVC(C=optimal_C, gamma=optimal_gamma)
history = svc.fit(X_train_scaled, y_train)
print("6.선형분류모델 : optimal C={:.12f} gamma={:.12f} 인 SVC의 훈련/테스트 정확도: {:.5f}/{:.5f}".
    format(optimal_C, optimal_gamma, svc.score(X_train_scaled, y_train), svc.score(X_test_scaled, y_test)))
plt.plot(gammaValues, train_scores, '-', label="훈련 정확도")
plt.plot(gammaValues, test_scores, '--', label="테스트 정확도")
plt.axvline(optimal_gamma, linestyle=':')
plt.xlabel("gamma")
plt.ylabel("정확도")
plt.legend()
plt.title('life 데이터셋에 각기 C={:.12f} 값 다른 gamma 을 사용하여 만든 SVC의 훈련/테스트 정확도'.format(optimal_C))
images.image.save_fig("6.20.life_SVC_gamma_score")  
plt.show()
