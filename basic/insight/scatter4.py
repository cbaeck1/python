# 산점도 행렬 (Scatter Plot Matrix)
# 예제로 사용할 데이터는 iris 데이터셋에 들어있는 4개의 연속형변수들인 'petal_length', 'petal_width', 'sepal_length', 'sepal_width' 입니다. 
# matplotlib 으로 산점도 행렬을 그리려면 코드가 너무 길어지고 가독성도 떨어지므로 추천하지 않으며, 
# seaborn 과 pandas, plotly 를 사용한 산점도 행렬
# (1) seaborn을 이용한 산점도 행렬 (scatterplot matrix by seaborn)
# (2) pandas를 이용한 산점도 행렬 (scatterplot matrix by pandas)
# (3) plotly를 이용한 산점도 행렬 (interactive scatterplot matrix by plotly)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 8] # setting figure size

# loading 'iris' dataset from seaborn
iris = sns.load_dataset('iris')

iris.info()
print("iris 크기: {}".format(iris.shape))
print("iris 5개: {}".format(iris.head()))

# (1) seaborn을 이용한 산점도 행렬 (scatterplot matrix by seaborn)
# default 설정을 사용하여 4개의 연속형 변수만을 가지고 그린 산점도 행렬입니다. 
# ('species' 범주형 변수는 알아서 무시해주니 참 편리합니다!)  코드도 간결하고 그래프도 깔끔하니 이뻐서 정말 마음에 듭니다! 
# 대각원소 자리에는 diag_kind='hist' 를 설정하여 각 변수별 히스토그램을 볼 수 있게 하였습니다. 

# scatterplot matrix with histogram only for continuous variables
sns.pairplot(iris, diag_kind='hist')
plt.show()

# 아래의 산점도 행렬에는 diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선을 볼 수 있게 하였으며, 
# hue='species'를 사용하여 'species' 종(setosa, versicolor, virginica) 별로 색깔을 다르게 표시하여 
# 추가적인 정보를 알 수 있도록 하였습니다. 
# 색깔은 palette 에 'bright', 'pastel', 'deep', 'muted', 'colorblind', 'dark' 중에서 가독성이 좋고 선호하는 색상으로 선택하면 됩니다. 

# Scatterplot matrix with different color by group and kde
sns.pairplot(iris, 
             diag_kind='kde',
             hue="species", 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
plt.show()

# (2) pandas를 이용한 산점도 행렬 (scatterplot matrix by pandas)
# 아래는 pandas.plotting 의 scatter_matrix() 함수를 사용하여 산점도 행렬을 그려본 것인데요, 
# 코드가 간결하긴 하지만 위의 seaborn 대비 그래프가 그리 아름답지는 않고 좀 투박합니다. 

# scatterplot matrix by pandas scatter_matrix()
from pandas.plotting import scatter_matrix
scatter_matrix(iris, 
               alpha=0.5, 
               figsize=(8, 8), 
               diagonal='kde')
plt.show()

# (3) plotly를 이용한 산점도 행렬 (interactive scatterplot matrix by plotly)
# plotly를 이용하면 분석가와 상호작용할 수 있는 역동적인 산점도 행렬 (interactive scatterplot matrix)을 만들 수 있습니다. 
# API 대신에 오프라인 모드(offline mode)에서 사용할 수 있도록 아래에 제시한 패키지들을 pip로 설치하고, import 해주어야 합니다. 

# import plotly standard
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Cufflinks wrapper on plotly
import cufflinks as cf
# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
# plotly + cufflinks in offline mode
from plotly.offline import iplot

cf.go_offline()
# set the global theme
cf.set_config_file(world_readable=True, theme='pearl', offline=True)

# plotly.offline의 iplot을 사용하여 오프라인 모드에서 산점도 행렬을 그린 결과입니다. 
# diag='histogram'으로 대각 행렬 위치에는 각 변수의 히스토그램을 그렸으며, 'scatter' (점 그림)와 'box' (박스 그림) 을 설정할 수도 있습니다. 
# 아래는 화면 캡펴한 이미지를 넣었는데요, jupyter notebook에서 보면 커서를 가져다데는 곳에 x, y 좌표 값이 실시간으로 화면에 INTERACTIVE하게 나타납니다. 
# hover 기능도 있어서 커서로 블록을 설정하면 블록에 해당하는 부분만 다시 산점도가 그려지기도 하며, file로 바로 다운로드도 가능합니다. 

fig = ff.create_scatterplotmatrix(
    iris[['petal_width', 'petal_length', 'sepal_width', 'sepal_length']],
    height=800,
    width=800, 
    diag='histogram') # scatter, histogram, box
iplot(fig) # offline mode

# 아래의 산점도 행렬에서는 대각행렬 위치에 '박스 그림 (diag='box')'을 제시하였고, 
# 산점도와 대각원소의 박스 그림을 index='species'를 사용하여 3개 종(setosa, versicolor, virginica) 별로  색깔을 다르게 구분해서 그려본 것입니다. 
# 아래 그림은 화면 캡쳐한 것이어서 interactive 하지 않은데요 (-_-;;;), jupyter notebook에서 실행해보면 커서를 위로 올려놓으면 데이터 값이 나오구요, 
# 줌 인/아웃, hover, 다운로드 등 interactive 한 시각화가 가능합니다. 

# scatterplot matrix by plotly with box plot at diagonal & different color by index(GROUP)
fig = ff.create_scatterplotmatrix(
    iris,
    height=800,
    width=800, 
    diag='box', # scatter, histogram, box
    index='species')
iplot(fig)
