import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 15. load_wine
from sklearn import datasets
wine = datasets.load_wine()

# print the names of the 13 features
print(wine['DESCR']+ "\n...")
print("wine.keys(): \n{}".format(wine.keys()))
print("데이터의 형태: {}".format(wine.data.shape))
# print the label type of wine(class_0, class_1, class_2)
print("특성 이름:\n{}".format(wine.feature_names))
print("Labels: ", wine.target_names)
print(wine.data, wine.target)
print(wine.target.mean(), wine.target.min(), np.percentile(wine.target,25), np.percentile(wine.target,75), wine.target.max())
print(wine.data[0:5], wine.target)


# 산점도 : 1개의 특성, 1개의 타겟(숫자)
plt.plot(wine.data[:, 0], wine.target, '.', 'MarkerSize', 2)
plt.xlabel("특성 1 : Alcohol")
plt.ylabel("Target : class")
plt.title("wine Scatter Plot")
images.image.save_fig("15.wine_Scatter")  
plt.show()

# 히스토그램 : 열의 이름은 wine.feature_names
# 사용할 특성의 갯수을 설정
nCase = 13
wine_df = pd.DataFrame(wine.data[:,:nCase], columns=wine.feature_names[:nCase])
# 데이터프레임을 사용해  특성별 Historgram
wine_df.plot.hist(bins=100, alpha=0.5)
plt.title("wine Histogram Plot")
images.image.save_fig("15.wine_Histogram")
plt.show() 

#  다른방법으로 pairplot
import seaborn as sns
wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_data.info()
print("wine_data 크기: {}".format(wine_data.shape))
print("wine_data 5개: {}".format(wine_data.head()))
wine_target = pd.DataFrame(wine.target, columns=['classType'])
wine_target.info()
print("wine_target 크기: {}".format(wine_target.shape))
print("wine_target 5개: {}".format(wine_target.head()))
wine2 = pd.merge(wine_data, wine_target, left_index=True, right_index=True)
print("wine2 크기: {}".format(wine2.shape))
print("wine2 5개: {}".format(wine2.head()))

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
sns.pairplot(wine2, 
             diag_kind='kde',
             hue='classType', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("15.Wine_Scatter_by_seaborn2")     
plt.show()


