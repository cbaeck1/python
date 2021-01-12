import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 3. 위스콘신 유방암 Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer라고 하겠습니다). 
# 각 종양은 양성benign(해롭지 않은 종양)과 악성malignant(암 종양)으로 레이블되어 있고, 
# 조직 데이터를 기반으로 종양이 악성인지를 예측할 수 있도록 학습하는 것이 과제
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer['DESCR']+ "\n...")
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))
print(cancer.data, cancer.target)
print(cancer.data[:,:2])


# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)으로
mglearn.discrete_scatter(cancer.data[:, 0], cancer.data[:, 1], cancer.target)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("mean radius")
plt.ylabel("mean texture")
plt.title("breast_cancer Scatter Plot")
images.image.save_fig("3.breast_cancer_Scatter")  
plt.show()

# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([cancer.data, X_train, X_test], [cancer.target, y_train, y_test], ['전체','X_train','X_test'], axes):
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title("{}".format(title))
  ax.set_xlabel("mean radius")
  ax.set_ylabel("mean texture")

axes[0].legend(loc=3)
images.image.save_fig("3.breast_cancer_scatter_compare")  
plt.show()

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 cancer.feature_names 에 있는 문자열을 사용합니다.
# 사용할 특성의 갯수을 설정
nCase = 29
breast_cancer_df= pd.DataFrame(X_train[:,:nCase], columns=cancer.feature_names[:nCase])
# 데이터프레임을 사용해  특성별 Historgram
breast_cancer_df.plot.hist(alpha=0.5, bins=100, figsize=(10, 10))
plt.title("breast_cancer Histogram Plot")
images.image.save_fig("3.breast_cancer_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
if nCase <= 10:
    pd.plotting.scatter_matrix(breast_cancer_df, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=20, alpha=.8, cmap=mglearn.cm3)
    plt.title("breast_cancer Scatter Plot")
    images.image.save_fig("3.breast_cancer_scatter_X_train")  
    plt.show()


# Scatterplot matrix with different color by group and kde
import seaborn as sns
cancer_data = pd.DataFrame(cancer['data'], columns=cancer.feature_names)
cancer_data.info()
print("cancer_data 크기: {}".format(cancer_data.shape))
print("cancer_data 5개: {}".format(cancer_data.head()))
cancer_target = pd.DataFrame(cancer['target'], columns=['targets'])
cancer_target.info()
print("cancer_target 크기: {}".format(cancer_target.shape))
print("cancer_target 5개: {}".format(cancer_target.head()))
# 사용할 특성의 갯수을 설정
nCase = 10
cancerDf = pd.merge(cancer_data.iloc[:,:nCase], cancer_target, left_index=True, right_index=True)
print("cancer 크기: {}".format(cancerDf.shape))
print("cancer 5개: {}".format(cancerDf.head()))

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
# hue='targets'를 사용하여 색깔을 다르게 표시
sns.pairplot(cancerDf, 
             diag_kind='kde',
             hue='targets', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("3.breast_cancer_Scatter_by_seaborn")     
plt.show()
