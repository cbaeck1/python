import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 9. 두 개의 클래스를 가진 2차원 데이터셋 make_moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:5], y[:5])

######################################################################### 
#  23. DBSCAN 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN() # eps기본값인 0.5
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('복잡한 모양의 클러스터를 DBSCAN 알고리즘')
images.image.save_fig("9.13moons_spiral_scatter_DBSCAN0.5")  
plt.show()

# eps기본값인 0.5 에서 예상한 클러스터 개수인 2개
# eps 0.7 로 변경 -> 클러스터를 1개로 만든다.
scaler07 = StandardScaler()
scaler07.fit(X)
X_scaled07 = scaler07.transform(X)
dbscan07 = DBSCAN(eps=0.7)
clusters = dbscan07.fit_predict(X_scaled07)

plt.scatter(X_scaled07[:, 0], X_scaled07[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('복잡한 모양의 클러스터를 DBSCAN(erp=0.7) 알고리즘')
images.image.save_fig("9.13moons_spiral_scatter_DBSCAN0.7")  
plt.show()

# 0.2로 내리면 클러스터를 8개로 만든다.
scaler02 = StandardScaler()
scaler02.fit(X)
X_scaled02 = scaler02.transform(X)
dbscan02 = DBSCAN(eps=0.2)
clusters = dbscan02.fit_predict(X_scaled02)

plt.scatter(X_scaled02[:, 0], X_scaled02[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('복잡한 모양의 클러스터를 DBSCAN(erp=0.2) 알고리즘')
image.save_fig("9.13moons_spiral_scatter_DBSCAN0.2")  
plt.show()


# 1. 타깃값으로 군집 평가하기 
# 군집 알고리즘의 결과를 실제 정답 클러스터와 비교하여 평가할 수 있는 지표
#  1. ARI (adjusted rand index)
#     ARI : 1(최적일 때)와 0(무작위로 분류될 때)
#  2. NMI (normalized mutual information)
# 

