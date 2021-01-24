import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import images.image

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
# 1. k-최근접 이웃 알고리즘 : 분류 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)

# 훈련 세트를 사용하여 분류 모델을 학습
clf.fit(X_train, y_train)
print("훈련 세트 정확도: {:.2f}".format(clf.score(X_train, y_train)))

# 예측
prediction = clf.predict(X_test)
print("테스트 세트 예측: {}".format(prediction))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, y_test)))

# [MinMaxScaler 를 사용하여 데이터 전처리 후 성능]
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X_train)
X_train_scaled = minmax_scaler.transform(X_train)
X_test_scaled = minmax_scaler.transform(X_test)
#print("X_train_scaled 크기: {}".format(X_train_scaled.shape))
#print(X_train_scaled)
#print("X_test_scaled 크기: {}".format(X_test_scaled.shape))

# 예측
prediction = clf.predict(X_train_scaled)
print("MinMaxScaler 테스트 세트 예측: {}".format(prediction))
print("MinMaxScaler 테스트 세트 정확도: {:.2f}".format(clf.score(X_test_scaled, y_test)))

# [StandardScaler 를 사용하여 데이터 전처리 후 성능]
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train_scaled_standard = standard_scaler.transform(X_train)
X_test_scaled_standard = standard_scaler.transform(X_test)
#print("X_train_scaled_standard 크기: {}".format(X_train_scaled_standard.shape))
#print(X_train_scaled_standard)
#print("X_test_scaled_standard 크기: {}".format(X_test_scaled_standard.shape))

# 예측
prediction = clf.predict(X_train_scaled_standard)
print("StandardScaler 테스트 세트 예측: {}".format(prediction))
print("StandardScaler 테스트 세트 정확도: {:.2f}".format(clf.score(X_test_scaled_standard, y_test)))

# n_neighbors 값이 각기 다른 최근접 이웃 모델이 만든 결정 경계
# 이웃의 수를 늘릴수록 결정 경계는 더 부드러워집니다
# 2개의 특성으로

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_life[:,:2], Y_life)
    mglearn.plots.plot_2d_separator(clf, X_life[:,:2], fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X_life[:, 0], X_life[:, 2], Y_life, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 매출_금액")
    ax.set_ylabel("특성 자본_금액")
axes[0].legend(loc=3)

images.image.save_fig("1.20.life_KNN_n_neighbors_1_3_9", "ml")  
plt.show()


# n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 21)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
images.image.save_fig("1.20.life_KNN_n_neighbors_1_20", "ml")  
plt.show()

# MinMaxScaler n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도 X_train_scaled, X_test_scaled
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 21)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train_scaled, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test_scaled, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
images.image.save_fig("1.20.MinMaxScaler_life_KNN_n_neighbors_1_20", "ml")  
plt.show()

# StandardScaler n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도 X_train_scaled_standard, X_test_scaled_standard
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 21)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train_scaled_standard, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test_scaled_standard, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
images.image.save_fig("1.20.StandardScaler_life_KNN_n_neighbors_1_20", "ml")  
plt.show()