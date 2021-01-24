import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import images.image

# 1. 붓꽃iris 데이터셋
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("타깃의 이름: {}".format(iris_dataset['target_names']))
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
print("data의 타입: {}".format(type(iris_dataset['data'])))
print("data의 크기: {}".format(iris_dataset['data'].shape))
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))

print("target의 타입: {}".format(type(iris_dataset['target'])))
print("target의 크기: {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))

# Scatterplot matrix with different color by group and kde
import seaborn as sns
iris_seaborn = sns.load_dataset('iris')
iris_seaborn.info()
print("iris 크기: {}".format(iris_seaborn.shape))
print("iris 5개: {}".format(iris_seaborn.head()))

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
# hue='species'를 사용하여 'species' 종(setosa, versicolor, virginica) 별로 색깔을 다르게 표시
sns.pairplot(iris_seaborn, 
             diag_kind='kde',
             hue='species', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("1.Iris_Scatter_by_seaborn1", "ml")     
plt.show()

#  다른방법으로 pairplot
iris_data = pd.DataFrame(iris_dataset['data'], columns=iris_dataset.feature_names)
iris_data.info()
print("iris_data 크기: {}".format(iris_data.shape))
print("iris_data 5개: {}".format(iris_data.head()))
iris_target = pd.DataFrame(iris_dataset['target'], columns=['species'])
iris_target.info()
print("iris_target 크기: {}".format(iris_target.shape))
print("iris_target 5개: {}".format(iris_target.head()))
iris = pd.merge(iris_data, iris_target, left_index=True, right_index=True)
print("iris 크기: {}".format(iris.shape))
print("iris 5개: {}".format(iris.head()))

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
# hue='species'를 사용하여 'species' 종(setosa, versicolor, virginica) 별로 색깔을 다르게 표시
sns.pairplot(iris, 
             diag_kind='kde',
             hue='species', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("1.Iris_Scatter_by_seaborn2", "ml")       
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names 에 있는 문자열을 사용합니다.
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe.info()
print("iris_dataframe 크기: {}".format(iris_dataframe.shape))
print("iris_dataframe 5개: {}".format(iris_dataframe.head()))

iris_dataframe.plot.hist(alpha=0.5)
plt.title("Iris Histogram Plot")
images.image.save_fig("1.Iris_Histogram", "ml")  
plt.show()

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
  hist_kwds={'bins': 20}, s=20, alpha=.8, cmap=mglearn.cm3)
plt.title("Iris Scatter Plot")
images.image.save_fig("1.Iris_Scatter", "ml")   
plt.show()

# seaborn 을 이용한 산점도 행렬
sns.pairplot(iris_dataframe, diag_kind='hist')
images.image.save_fig("1.Iris_Scatter_by_seaborn3", "ml")                
plt.show()


