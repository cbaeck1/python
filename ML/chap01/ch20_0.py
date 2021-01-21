import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import images.image

# 20. life

# 1.1 판다스로 데이터프레임 만들기
dataframe = pd.read_csv('d:/data/trans서울중구_org.csv')
print(dataframe.head(), dataframe.shape)

X_dataframe = dataframe.loc[:,'매출_금액':'주소']
print(X_dataframe.head(), X_dataframe.shape)

Y_dataframe = dataframe['TARGET']
# YN = pd.Categorical(Y_dataframe)
# Y_dataframe = YN.codes # numpy.array
# Y_dataframe = Y_dataframe.apply(pd.to_numeric)
# print(Y_dataframe.head(), Y_dataframe.shape)

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)으로
mglearn.discrete_scatter(X_dataframe.iloc[:, 0], X_dataframe.iloc[:, 4], Y_dataframe)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("매출_금액")
plt.ylabel("종업원_수")
plt.title("life Scatter Plot")
images.image.save_fig("20.1.life_Scatter")  
plt.show()

# 산점도를 그립니다. 1개의 특성과 1개의 타켓(2개의 값)으로
mglearn.discrete_scatter(X_dataframe.iloc[:, 0], Y_dataframe, Y_dataframe)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("매출_금액")
plt.ylabel("TARGET")
plt.title("life Scatter Plot")
images.image.save_fig("20.2.life_Scatter")  
plt.show()

# Historam 그립니다. 
# bins=5, ## 몇 개의 바구니로 구분
# density=True, ## ytick을 퍼센트비율로 표현해줌 
# cumulative=False, ## 누적으로 표현하고 싶을 때는 True
# histtype='bar',  ## 타입. or step으로 하면 모양이 바뀜.  
# orientation='vertical', ## or horizontal
# rwidth=0.8, ## 1.0일 경우, 꽉 채움 작아질수록 간격이 생김
# color='hotpink', ## bar 색깔
for i, col in enumerate(X_dataframe.loc[:,'매출_금액':'산재_비율']):
    #print(i, col)
    plt.title("life_Histogram")
    plt.xlabel(col)
    images.image.save_fig("20."+str(i)+"." + col + " Histogram")  
    plt.hist(X_dataframe.iloc[:, i])
    #plt.show()

# 산점도를 그립니다. 1개의 특성과 1개의 타켓으로
for i, col in enumerate(X_dataframe.loc[:,'매출_금액':'산재_비율']):
    mglearn.discrete_scatter(X_dataframe.iloc[:, i], Y_dataframe, Y_dataframe)
    plt.title("life_Scatter")
    plt.xlabel(col)
    plt.ylabel("TARGET")
    plt.legend(["클래스 0", "클래스 1"], loc=4)
    images.image.save_fig("20."+str(i)+"." + col + " Scatter")  
    #plt.show()

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)으로
for i, col in enumerate(X_dataframe.loc[:,'매출_금액':'산재_비율']):
    for j, row in enumerate(X_dataframe.loc[:,'매출_금액':'산재_비율']):
        if i == j: 
            continue
        mglearn.discrete_scatter(X_dataframe.iloc[:, i], X_dataframe.iloc[:, j], Y_dataframe)
        plt.title("life_Scatter")
        plt.xlabel(col)
        plt.ylabel(row)
        plt.legend(["클래스 0", "클래스 1"], loc=4)
        images.image.save_fig("20."+str(i)+"." +str(j)+"." + col + " Scatter")  
        #plt.show()

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataframe, test_size=0.2)
X_train = train.loc[:,'매출_금액':'주소']
y_train = train['TARGET']
X_test = test.loc[:,'매출_금액':'주소']
y_test = test['TARGET']
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
train, val = train_test_split(train, test_size=0.2)
print(len(val), '검증 샘플')

# 산점도 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([dataframe, X_train, X_test], [Y_dataframe, y_train, y_test], ['전체','X_train','X_test'], axes):
    mglearn.discrete_scatter(X.iloc[:, 0], X.iloc[:, 2], y, ax=ax)
    ax.set_title("{}".format(title))
    ax.set_xlabel("매출_금액")
    ax.set_ylabel("자본_금액")

axes[0].legend(loc=3)
images.image.save_fig("20.life_scatter_compare")  
plt.show()

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 cancer.feature_names 에 있는 문자열을 사용합니다.
# 사용할 특성의 갯수을 설정
#nCase = 12
#breast_cancer_df= pd.DataFrame(X_train[:,:nCase], columns=cancer.feature_names[:nCase])
# 데이터프레임을 사용해  특성별 Historgram
dataframe.plot.hist(alpha=0.5, bins=100, figsize=(10, 10))
plt.title("life Histogram Plot")
images.image.save_fig("20.life_Histogram")
plt.show() 

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
#if nCase <= 10:
pd.plotting.scatter_matrix(X_dataframe, c=Y_dataframe, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=20, alpha=.8, cmap=mglearn.cm3)
plt.title("life Scatter Plot")
images.image.save_fig("20.life_scatter_X_dataframe")  
plt.show()

# Scatterplot matrix with different color by group and kde
import seaborn as sns
# 사용할 특성의 갯수을 설정
nCase = 10
lifeDf = pd.merge(dataframe.iloc[:,:nCase], dataframe['TARGET'], left_index=True, right_index=True)
print("dataframe 크기: {}".format(lifeDf.shape))
print("dataframe 5개: {}".format(lifeDf.head()))
# heartTarget = pd.DataFrame(Y_dataframe, columns=['target'])

# diag_kind='kde' 를 사용하여 각 변수별 커널밀도추정곡선
# hue='targets'를 사용하여 색깔을 다르게 표시
sns.pairplot(lifeDf, 
             diag_kind='kde',
             hue='TARGET', 
             palette='bright') # pastel, bright, deep, muted, colorblind, dark
images.image.save_fig("20.life_Scatter_by_seaborn")     
plt.show()
