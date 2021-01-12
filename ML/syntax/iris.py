import numpy as np
import pandas as pd
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

s = open('g:\work\workspace\ml\syntax\iris.csv').readline()
#header = [i.strip('"') for i in s.strip().split(',')][:-1]
header = s.strip().split(',')[:-1]
print("header", header)

labels = ['setosa', 'versicolor', 'virginica']
iris = np.loadtxt('g:\work\workspace\ml\syntax\iris.csv', delimiter=',', skiprows=1, converters={4: lambda s: labels.index(s.decode())})
print(iris.shape, iris[:5])

X = iris[:,:4]
y = iris[:,4]

print(X.shape, y.shape)
print(X.dtype, y.dtype)
print(type(X), type(y))
print(X[:5], y)
print(type(X[:,0]), type(y))

n = len(X)
print(n)
# s : 마커의 크기
# c : 마커의 색상
S = np.abs(np.random.randn(n))*100
C = np.random.randint(0, n, n)

# scatter
plt.figure(figsize=[12,8])

position = 1
for row in range(4):
    for col in range(4):
        plt.subplot(4,4,position)
        plt.scatter(X[:,row], X[:,col], c=None, s=None, alpha=0.8)
        plt.title("Corr "+header[row]+" vs "+header[col], fontsize=10)
        position = position + 1
save_fig("Correlation_Coefficient")
plt.show()

'''
plt.subplot(4,4,1)
plt.scatter(X[:,0], X[:,0], c=None, s=None)
plt.title("Corr "+header[0]+" vs "+header[0], fontsize=10)
plt.subplot(4,4,2)
plt.scatter(X[:,0], X[:,1], c=None, s=None)
plt.title("Corr "+header[0]+" vs "+header[1], fontsize=10)
plt.subplot(4,4,3)
plt.scatter(X[:,0], X[:,2], c=None, s=None)
plt.title("Corr "+header[0]+" vs "+header[2], fontsize=10)
plt.subplot(4,4,4)
plt.scatter(X[:,0], X[:,3], c=None, s=None)
plt.title("Corr  "+header[0]+" vs "+header[3], fontsize=10)
'''



# 피어슨 상관계수 (Correlation Coefficient)
corr = np.corrcoef(X.T)
print(X.T)
print("Correlation Coefficient", corr)

corr2 = np.corrcoef(X)
print("Correlation Coefficient", corr2)

plt.imshow(corr, interpolation='none', vmin=-1, vmax=1, cmap='spring')
plt.xticks(range(4), header, rotation=90)
plt.yticks(range(4), header)
plt.colorbar()
save_fig("colorbar")
plt.show()

plt.boxplot(X)
plt.xticks(range(1,5), header, rotation=90)
save_fig("boxplot")
plt.show()

plt.figure(figsize=[12,8])
for col in range(4):
    plt.subplot(2,2,col+1)
    # plt.scatter(X[:,col], y, c=y, s=30, alpha=0.2)
    plt.scatter(X[:,col], y + np.random.normal(0,0.03,size=len(y)), c=y, s=30, alpha=0.3)
    plt.yticks([0,1,2], ['Setosa', 'Versicolor', 'Virginica'], rotation=90)
    plt.title(header[col], fontsize=15)

save_fig("scatter")
plt.show()

iris_df = pd.DataFrame(X, columns=header)
print(iris_df.info())
print(iris_df.describe())
iris_df.plot(kind='hist', alpha=0.3)
save_fig("iris_df.hist")
plt.show()

pd.plotting.scatter_matrix(iris_df, c=y, s=60, alpha=0.8, figsize=[12,12])
save_fig("iris_df.scatter_matrix")
plt.show()

ct, bins = np.histogram(X[:,0], 20)
plt.hist(X[:,0][y==0], bins=bins, alpha=0.3)
plt.hist(X[:,0][y==1], bins=bins, alpha=0.3)
plt.hist(X[:,0][y==2], bins=bins, alpha=0.3)
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
save_fig("ct.histogram")
plt.show()

titles = ['Setosa', 'Versicolor', 'Virginica']
plt.figure(figsize=[12,8])

for col in range(4):
    plt.subplot(2,2,col+1)
    plt.title(header[col], fontsize=15)
    ct, bins = np.histogram(X[:,col], 20)
    plt.hist(X[:,col][y==0], bins=bins, alpha=0.3)
    plt.hist(X[:,col][y==1], bins=bins, alpha=0.3)
    plt.hist(X[:,col][y==2], bins=bins, alpha=0.3)
    plt.ylim(0,40)
    if(col==0): plt.legend(titles)

save_fig("ct.histogram2")
plt.show()

N=30

plt.plot(X[:N].T, 'r-', alpha=0.3)
plt.plot(X[50:50+N].T, 'b-', alpha=0.3)
plt.plot(X[100:100+N].T, 'g-', alpha=0.3)
plt.xticks(range(4), header)

save_fig("N30.rgb")
plt.show()

Xs = [X[y==0], X[y==1], X[y==2]]
plt.figure(figsize=[8,12])
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(X[y==i])
    plt.title(titles[i], fontsize=15)
    plt.ylim(0,10)
    if i==2: plt.xlabel('samples', fontsize=15)
    if i==0: plt.legend(header)

save_fig("ylim.10")
plt.show()

col1 = 0
col2 = 1

plt.scatter(X[:,col1], X[:,col2], c=y, s=60)
plt.colorbar(shrink=0.5)
plt.xlabel(header[col1])
plt.ylabel(header[col2])

save_fig("shrink_colorbar")
plt.show()







