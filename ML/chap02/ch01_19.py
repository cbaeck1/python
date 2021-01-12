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
X_dataframe = dataframe.loc[:,'age':'thal']
X_dataframe.thal = pd.Categorical(X_dataframe.thal)
X_dataframe['thal'] = X_dataframe.thal.cat.codes
X_dataframe = X_dataframe.apply(pd.to_numeric)
print(X_dataframe.head(), X_dataframe.shape, X_dataframe.info())

# numpy array
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X_dataframe)
X_heart = imputer.transform(X_dataframe)

Y_heart = dataframe['target']
print(Y_heart.head(), Y_heart.shape)

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, stratify=Y_heart, random_state=66)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))

'''
# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
train, test = train_test_split(dataframe, test_size=0.2)
X_train = train.loc[:,'age':'thal'].to_numpy()
y_train = train['target'].to_numpy()
X_test = test.loc[:,'age':'thal'].to_numpy()
y_test = test['target'].to_numpy()
train, val = train_test_split(train, test_size=0.2)
print(len(val), '검증 샘플')
'''

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

# n_neighbors 값이 각기 다른 최근접 이웃 모델이 만든 결정 경계
# 이웃의 수를 늘릴수록 결정 경계는 더 부드러워집니다
# 2개의 특성으로

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit 메서드는 self 객체를 반환합니다.
    # 그래서 객체 생성과 fit 메서드를 한 줄에 쓸 수 있습니다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_heart[:,:2], Y_heart)
    mglearn.plots.plot_2d_separator(clf, X_heart[:,:2], fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X_heart[:, 0], X_heart[:, 3], Y_heart, ax=ax)
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 age")
    ax.set_ylabel("특성 trestbps")
axes[0].legend(loc=3)

images.image.save_fig("1.19.heart_KNN_n_neighbors_1_3_9")  
plt.show()


# n_neighbors 변화에 따른 훈련 정확도와 테스트 정확도
training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 101)

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
images.image.save_fig("1.19.heart_KNN_n_neighbors_1_10")  
plt.show()






