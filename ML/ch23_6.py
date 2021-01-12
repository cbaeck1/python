import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 6. 세 개의 클래스를 가진 간단한 blobs 데이터셋
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=0, n_samples=12)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

########################################################################
#  23. DBSCAN 
#  모든 포인트에 잡음 포인트를 의미하는 -1레이블이 할당
#  작은 샘플 데이터셋(12개)에 적합하지 않은 eps(eps기본값인 0.5)와 min_samples기본값 때문
from sklearn.cluster import DBSCAN
dbscan = DBSCAN() # eps기본값인 0.5
clusters = dbscan.fit_predict(X)
print("클러스터 레이블 : \n{}".format(clusters))

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title("blobs 12개 샘플 DBSCAN plot")
images.image.save_fig("6.13blobs_sample12_DBSCAN_plot")     
plt.show()

# n_samples=200
X, y = make_blobs(random_state=0, n_samples=200)
#dbscan = DBSCAN() # eps기본값인 0.5
clusters200 = dbscan.fit_predict(X)
print("클러스터 레이블 : \n{}".format(clusters200))

plt.scatter(X[:, 0], X[:, 1], c=clusters200, cmap=mglearn.cm2, s=60, edgecolors='black')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title("blobs 200개 샘플 DBSCAN plot")
images.image.save_fig("6.13blobs_sample200_DBSCAN_plot")     
plt.show()

