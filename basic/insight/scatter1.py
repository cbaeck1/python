# (1) matplotlib을 사용한 산점도 (scatter plot by matplotlib)
# (2) seaborn을 사용한 산점도 (scatter plot by seaborn)
# (3) pandas를 사용한 산점도 (scatter plot by pandas)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = [12, 8] # setting figure size

# loading 'iris' dataset from seaborn : DataFrame
iris = sns.load_dataset('iris')

print("iris.keys(): \n{}".format(iris.keys()))
print("iris 데이터의 형태: {}".format(iris.shape))
print(iris)

# (1) matplotlib을 사용한 산점도 (scatter plot by matplotlib)

plt.plot('petal_length',  # x
         'petal_width',  # y
         data=iris, 
         linestyle='none', 
         marker='o', 
         markersize=10,
         color='blue', 
         alpha=0.5)

plt.title('Scatter Plot of iris by matplotlib', fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()


# 산점도에 X, Y 좌표를 사용하여 직사각형(rectangle), 원(circle), 직선(line)을 추가하여 보겠습니다. 
# 먼저 matplotlib.patches 를 importing 해주어야 하고, 산점도를 그린 다음에, add_patch() 함수를 사용하여 직사각형, 원, 직선을 추가합니다. 

# adding a rectangle, a circle
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

# (0) scatter plot
ax1.plot('petal_length', 'petal_width', data=iris, linestyle='none', marker='o')
# (1) adding a rectangle
ax1.add_patch(
    patches.Rectangle(
        (3, 1), # (x, y)
        2, # width
        1, # height
        alpha=0.2, 
        facecolor="blue", 
        edgecolor="black", 
        linewidth=2, 
        linestyle="solid", 
        angle=-10))
# (2) adding a circle
ax1.add_patch(
    patches.Circle(
        (1.5, 0.25), # (x, y)
        0.5, # radius
        alpha=0.2, 
        facecolor="red", 
        edgecolor="black", 
        linewidth=2, 
        linestyle='solid'))
# (3) adding a line
plt.plot([4, 6], [2.2, 1.1], color="green", lw=4, linestyle='solid')
plt.title("Adding a Rectangle, a Circle and a Line", fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()

# (2) seaborn을 사용한 산점도 (scatter plot by seaborn)
# seaborn 패키지의 (a) regplot() 함수와 (b) scatterplot() 함수를 사용해서 산점도를 그릴 수 있습니다.
# (a) regplot() 함수를 사용한 산점도
# 선형회귀 적합 선을 포함시키지 않으려면 fit_reg=False 를 설정해주면 됩니다. 

# Basic Scatter Plot by seaborn
sns.regplot(x=iris['petal_length'], 
           y=iris['petal_width'], 
           fit_reg=False) # no regression line
plt.title('Scatter Plot of iris by regplot()', fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()

# 두 연속형 변수 간의 선형회귀 적합선을 산점도에 포함시키려면 fit_reg=True 를 설정해주면 됩니다. 
# (defalt 이므로 별도로 표기를 해주지 않아도 회귀적합선이 추가됩니다)

# Scatter Plot with regression line by seaborn regplot()
sns.regplot(x=iris['petal_length'], 
           y=iris['petal_width'], 
           fit_reg=True) # default
plt.title('Scatter Plot with Regression Line by regplot()', fontsize=20)
plt.show()

# X축과 Y축의 특정 값의 조건을 기준으로 산점도 marker의 색깔을 다르게 해보겠습니다. 
# 'petal length' > 2.5 & 'petal width' > 0.8 이면 '빨간색', 그 이외는 '파란색'으로 설정을 해보겠습니다. 
# 조건에 맞게 'color'라는 새로운 변수를 생성한 후에, scatter_kws={'facecolors': iris_df['color']}) 로 조건별 색을 설정하는 방법을 사용하였습니다. 

# Control color of each marker based on X and Y values
iris_df = iris.copy()
# Adding a 'color' column based on x and y values
cutoff = (iris_df['petal_length']>2.5) & (iris_df['petal_width'] > 0.8)
iris_df['color'] = np.where(cutoff==True, "red", "blue")

# Scatter Plot with different colors based on X and Y values
sns.regplot(x=iris['petal_length'], 
           y=iris['petal_width'], 
           fit_reg=False, 
           scatter_kws={'facecolors': iris_df['color']}) # marker color
plt.title('Scatter Plot with different colors by X & Y values', fontsize=20)
plt.show()

# (b) scatterplot() 함수를 사용한 산점도
# scatter plot by seaborn scatterplot()
ax = sns.scatterplot(x='petal_length', 
                     y='petal_width', 
                     alpha=0.5,
                     data=iris)
plt.title('Scatter Plot by seaborn', fontsize=20)
plt.show()


# (3) pandas를 사용한 산점도 (scatter plot by pandas)

iris.plot.scatter(x='petal_length', 
                  y='petal_width', 
                  s=50, # marker size
                  c='blue', 
                  alpha=0.5)
plt.title('Scatter Plot of iris by pandas', fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()
