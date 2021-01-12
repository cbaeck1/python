import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 4. 회귀 분석용 실제 데이터셋으로는 보스턴 주택가격 Boston Housing 데이터셋을 사용
# 이 데이터셋으로 할 작업은 범죄율, 찰스강 인접도, 고속도로 접근성 등의 정보를 이용해 
# 1970년대 보스턴 주변의 주택 평균 가격을 예측하는 것입니다. 
# 이 데이터셋에는 데이터 506개와 특성 13개가 있습니다
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['DESCR']+ "\n...")
print("boston.keys(): \n{}".format(boston.keys()))
print("데이터의 형태: {}".format(boston.data.shape))
print("특성 이름:\n{}".format(boston.feature_names))
print(boston.data, boston.target)
print(boston.target.mean(), boston.target.min(), np.percentile(boston.target,25), np.percentile(boston.target,75), boston.target.max())
print(boston.data[:,:2])

# 산점도 : 1개의 특성, 1개의 타겟(숫자)
plt.plot(boston.data[:, 0], boston.target, '.', 'MarkerSize', 2)
plt.ylim(0, 60)
plt.xlabel("특성 CRIM : per capita crime rate by town")
plt.ylabel("Target : house-price")
plt.title("boston Scatter Plot")
images.image.save_fig("4.boston_Scatter")  
plt.show()

# 히스토그램 : 열의 이름은 boston.feature_names
# 사용할 특성의 갯수을 설정
nCase = 13
boston_df = pd.DataFrame(boston.data[:,:nCase], columns=boston.feature_names[:nCase])
# 데이터프레임을 사용해  특성별 Historgram
boston_df.plot.hist(bins=100, alpha=0.5)
plt.title("boston Histogram Plot")
images.image.save_fig("4.boston_Histogram")
plt.show() 

#  다른방법으로 pairplot
import seaborn as sns
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_data.info()
print("boston_data 크기: {}".format(boston_data.shape))
print("boston_data 5개: {}".format(boston_data.head()))
# target을 세개의 값으로 변경
y_bin = np.array([0 if i < 17.0 else (1 if i < 25.0 else 2) for i in boston.target])
boston_target = pd.DataFrame(y_bin, columns=['priceType'])
boston_target.info()
print("boston_target 크기: {}".format(boston_target.shape))
print("boston_target 5개: {}".format(boston_target.head()))
boston = pd.merge(boston_data, boston_target, left_index=True, right_index=True)
print("boston 크기: {}".format(boston.shape))
print("boston 5개: {}".format(boston.head()))

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
# hue='species'를 사용하여 'species' 종(setosa, versicolor, virginica) 별로 색깔을 다르게 표시
sns.pairplot(boston, 
             diag_kind='kde',
             hue='priceType', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("4.Boston_Scatter_by_seaborn2")     
plt.show()


# 특성 공학 feature engineering : load_extended_boston
# 13개의 원래 특성에 13개에서 2개씩 (중복을 포함해) 짝지은 91개의 특성을 더해 총 104개가 됩니다.
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
# print(X, y)
print(X[:, 0], X[:, 1], y)

# 훈련 세트, 테스트 세트 random_state=66
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
print("X_train 타입: {}".format(type(X_train)))
print("y_train 타입: {}".format(type(y_train)))
print(X_train[:, 0], X_train[:, 1], y_train)

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 range로 표현
# 사용할 특성의 갯수을 설정
nCase = 4
extended_boston_df = pd.DataFrame(X_train[:,:nCase], columns=range(nCase))
# 데이터프레임을 사용해  특성별 Historgram
extended_boston_df.plot.hist(alpha=0.5)
plt.title("extended_boston_df Histogram Plot")
images.image.save_fig("4.extended_boston_df_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
if nCase <= 10:
    pd.plotting.scatter_matrix(extended_boston_df, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=2, alpha=.8, cmap=mglearn.cm3)
    plt.title("extended_boston_df Scatter Plot")
    images.image.save_fig("4.extended_boston_df_Scatter")  
    plt.show()


    

# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([X, X_train, X_test], [y, y_train, y_test], ['전체','X_train','X_test'], axes):
  # 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
  # target을 세개의 값으로 변경
  y_bin = np.array([0 if i < 17.0 else (1 if i < 25.0 else 2) for i in y])  
  mglearn.discrete_scatter(X[:, 5], X[:, 10], y_bin, ax=ax)
  ax.set_title("{}".format(title))
  ax.set_xlabel("average number of rooms per dwelling")
  ax.set_ylabel("TAX")

axes[0].legend(loc=3)
images.image.save_fig("4.boston_scatter_compare")  
plt.show()

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 range로 표현
# 사용할 특성의 갯수을 설정
nStart = 20
nCase = 10
extended_boston_df = pd.DataFrame(X_train[:,nStart:nStart+nCase], columns=range(nCase))
# 데이터프레임을 사용해  특성별 Historgram
extended_boston_df.plot.hist(bins=100, alpha=0.5)
plt.title("extended_boston Histogram Plot")
images.image.save_fig("4.extended_boston_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
y_bin = np.array([0 if i < 17.0 else (1 if i < 25.0 else 2) for i in y_train])  
if nCase <= 10:
  pd.plotting.scatter_matrix(extended_boston_df, c=y_bin, figsize=(15, 15), marker='o',
  hist_kwds={'bins': 20}, s=2, alpha=.8, cmap=mglearn.cm3)
  #plt.title("extended_boston Scatter Plot")
  images.image.save_fig("4.extended_boston_Scatter")  
  plt.show()

