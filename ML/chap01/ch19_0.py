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

X_dafaframe = dataframe.loc[:,'age':'thal']
Y_dataframe = dataframe['target']
print(X_dafaframe.head(), X_dafaframe.shape)
print(Y_dataframe.head(), Y_dataframe.shape)

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)으로
mglearn.discrete_scatter(X_dafaframe.iloc[:, 0], X_dafaframe.iloc[:, 3], Y_dataframe)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("age")
plt.ylabel("trestbps")
plt.title("heart Scatter Plot")
images.image.save_fig("19.heart_Scatter")  
plt.show()

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataframe, test_size=0.2)
X_train = train.loc[:,'age':'thal']
y_train = train['target']
X_test = test.loc[:,'age':'thal']
y_test = test['target']
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
train, val = train_test_split(train, test_size=0.2)
print(len(val), '검증 샘플')

# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([dataframe, X_train, X_test], [Y_dataframe, y_train, y_test], ['전체','X_train','X_test'], axes):
  mglearn.discrete_scatter(X.iloc[:, 0], X.iloc[:, 3], y, ax=ax)
  ax.set_title("{}".format(title))
  ax.set_xlabel("age")
  ax.set_ylabel("trestbps")

axes[0].legend(loc=3)
images.image.save_fig("19.heart_scatter_compare")  
plt.show()

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 cancer.feature_names 에 있는 문자열을 사용합니다.
# 사용할 특성의 갯수을 설정
#nCase = 12
#breast_cancer_df= pd.DataFrame(X_train[:,:nCase], columns=cancer.feature_names[:nCase])
# 데이터프레임을 사용해  특성별 Historgram
dataframe.plot.hist(alpha=0.5, bins=100, figsize=(10, 10))
plt.title("breast_cancer Histogram Plot")
images.image.save_fig("19.heart_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
#if nCase <= 10:
pd.plotting.scatter_matrix(X_dafaframe, c=Y_dataframe, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=20, alpha=.8, cmap=mglearn.cm3)
plt.title("heart Scatter Plot")
images.image.save_fig("19.heart_scatter_X_train")  
plt.show()

# Scatterplot matrix with different color by group and kde
import seaborn as sns
# 사용할 특성의 갯수을 설정
nCase = 10
heartDf = pd.merge(dataframe.iloc[:,:nCase], Y_dataframe, left_index=True, right_index=True)
print("heartDf 크기: {}".format(heartDf.shape))
print("heartDf 5개: {}".format(heartDf.head()))
# heartTarget = pd.DataFrame(Y_dataframe, columns=['target'])

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
# hue='targets'를 사용하여 색깔을 다르게 표시
sns.pairplot(heartDf, 
             diag_kind='kde',
             hue='target', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("19.heart_Scatter_by_seaborn")     
plt.show()
