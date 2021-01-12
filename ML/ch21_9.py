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
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:5], y[:5])

########################################################################
#  21. k-평균 군집
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# 클러스터 할당과 클러스터 중심 나타내기
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', 
    c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2, edgecolors='k')
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.title('복잡한 모양의 클러스터를 구분하지 못하는 k-평균 알고리즘')
images.image.save_fig("9.11 moons_scatter_kmeans_fail")  
plt.show()

# 복잡한 형태의 데이터셋을 다루기 위해 많은 클러스터를 사용한 k-평균
X_spiral, y_spiral = make_moons(n_samples=200, noise=0.05, random_state=0)
print("X_spiral.shape: {}".format(X_spiral.shape))
print("y_spiral.shape: {}".format(y_spiral.shape))
print(X_spiral[:5], y_spiral[:5])

#
kmeans10 = KMeans(n_clusters=10, random_state=0)
kmeans10.fit(X_spiral)
y_spiral_pred = kmeans10.predict(X_spiral)
print("클러스터 레이블:\n{}".format(y_spiral_pred))

plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral_pred, s=60, cmap='Paired', edgecolors='black')
plt.scatter(kmeans10.cluster_centers_[:, 0], kmeans10.cluster_centers_[:, 1], 
    s=60, marker='^', c=range(kmeans10.n_clusters), linewidth=2, cmap='Paired', edgecolor='black')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('복잡한 모양의 클러스터를 구분하지 못하는 k-평균 알고리즘')
images.image.save_fig("9.11 moons_spiral_scatter_kmeans_fail")  
plt.show()


