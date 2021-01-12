# 4개 변수로 점의 크기와 색을 다르게 산점도 그리기 (Scatter plot with different size, color)
# (1) matplotlib에 의한 4개 연속형 변수를 사용한 산점도 (X축, Y축, 색, 크기)
# (2) seaborn에 의한 4개 연속형 변수를 사용한 산점도 (X축, Y축, 색, 크기)
# (3) pandas에 의한 4개 연속형 변수를 사용한 산점도 (X축, Y축, 색, 크기)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 8] # setting figure size

# loading 'iris' dataset from seaborn
iris = sns.load_dataset('iris')

# (1) matplotlib에 의한 4개 연속형 변수를 사용한 산점도 (X축, Y축, 색, 크기)
# plt.scatter() 함수를 사용하며, 점의 크기는 s, 점의 색깔은 c 에 변수를 할당해주면 됩니다. 

# 4 dimensional scatter plot with different size & color
plt.scatter(iris.sepal_length, # x
           iris.sepal_width, # y
           alpha=0.2, 
           s=200*iris.petal_width, # marker size
           c=iris.petal_length, # marker color
           cmap='viridis')
plt.title('Scatter Plot with Size(Petal Width) & Color(Petal Length)', fontsize=14)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.colorbar()
plt.show()

# 점(marker)의 모양을 네모로 바꾸고 싶으면 marker='s' 로 설정해주면 됩니다. 
# 4 dimensional scatter plot with different size & color
plt.scatter(iris.sepal_length, # x
           iris.sepal_width, # y
           alpha=0.2, 
           s=200*iris.petal_width, # marker size
           c=iris.petal_length, # marker color
           cmap='viridis', 
           marker = 's') # square shape
plt.title('Size(Petal Width) & Color(Petal Length) with Square Marker', fontsize=14)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.colorbar()
plt.show()


# (2) seaborn에 의한 4개 연속형 변수를 사용한 산점도 (X축, Y축, 색, 크기)
# seaborn 의 산점도 코드는 깔끔하고 이해하기에 쉬으며, 범례도 잘 알아서 색깔과 크기를 표시해주는지라 무척 편리합니다. 

# 4 dimensional scatter plot by seaborn
sns.scatterplot(x='sepal_length', 
                y='sepal_width', 
                hue='petal_length',
                size='petal_width',
                data=iris)
plt.show()


# (3) pandas에 의한 4개 연속형 변수를 사용한 산점도 (X축, Y축, 색, 크기)
# pandas의 DataFrame에 plot(kind='scatter') 로 해서 color=iris['petal_length']로 색깔을 설정, 
# s=iris['petal_width'] 로 크기를 설정해주면 됩니다. pandas 산점도 코드도 깔끔하고 이해하기 쉽긴 한데요, 범례 추가하기가 쉽지가 않군요. ^^; 

iris.plot(kind='scatter', 
          x='sepal_length', 
          y='sepal_width', 
          color=iris['petal_length'],
          s=iris['petal_width']*100)
plt.title('Size(Petal Width) & Color(Petal Length) with Square Marker', fontsize=14)
plt.show()




