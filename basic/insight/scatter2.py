# 그룹별 산점도 (Scatter Plot by Groups)
# 산점도를 그리는데 사용할 데이터는 iris 로서, 'petal length'와 'petal width'의 연속형 변수에 대해서 
# 'species' 그룹별로 점의 색깔과 모양을 다르게 설정
# 참고로 species 에는 setosa 50개, versicolor 50개, virginica 50개씩의 관측치
#  (1) matplotlib 으로 그룹별 산점도 그리기 (scatter plot by groups via matplotlib)
#  (2) seaborn 으로 그룹별 산점도 그리기 (scatter plot by groups via seaborn)
#  (3) pandas로 그룹별 산점도 그리기 (scatter plot by groups via pandas)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 8] # setting figure size

# loading 'iris' dataset from seaborn
iris = sns.load_dataset('iris')

# (1) matplotlib 으로 그룹별 산점도 그리기 (scatter plot by groups via matplotlib)

# Scatter plot with a different color by groups
groups = iris.groupby('species')
print(groups.count())
print(groups.head())

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.petal_length, 
            group.petal_width, 
            marker='o', 
            linestyle='',
            label=name)
ax.legend(fontsize=12, loc='upper left') # legend position
plt.title('Scatter Plot of iris by matplotlib', fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()

# (2) seaborn 으로 그룹별 산점도 그리기 (scatter plot by groups via seaborn)
# 코드가 깔끔하고 가독성이 좋으며, 산점도 그래프도 보기에 참 좋습니다. 

# Scatter plot by Groups
sns.scatterplot(x='petal_length', 
                y='petal_width', 
                hue='species', # different colors by group
                style='species', # different shapes by group
                s=100, # marker size
                data=iris)
plt.show()

# (3) pandas로 그룹별 산점도 그리기 (scatter plot by groups via pandas)

# adding 'color' column
iris['color'] = np.where(iris.species == 'setosa', 'red', 
                         np.where(iris.species =='versicolor', 
                         'green', 'blue'))
# scatter plot
iris.plot(kind='scatter',
          x='petal_length', 
          y='petal_width', 
          s=50, # marker size
          c=iris['color']) # marker color by group
plt.title('Scatter Plot of iris by pandas', fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()

