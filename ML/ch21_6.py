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
X, y = make_blobs(random_state=1)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

########################################################################
#  21. k-평균 군집
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("클러스터 레이블:\n{}".format(kmeans.labels_))
print("클러스터 레이블:\n{}".format(kmeans.predict(X)))

# 군집은 각 데이터 포인트가 레이블을 가진다는 면에서 분류와 비슷하게 보인다.
# 그러나 정답을 모르고 있으며, 레이블 자체에 어떤 의미가 있지는 않는다.
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], 
    markers='^', markeredgewidth=2)
plt.title('세 개의 클래스를 가진 2차원 데이터셋 k-평균 산점도')
images.image.save_fig("6.11 blobs_Scatter")  
plt.show()    


# 두개의 클러스터 중심
kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(X)
assignments2 = kmeans2.labels_

# 다섯개의 클러스터 중심
kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(X)
assignments5 = kmeans5.labels_

# 세 개의 클래스를 가진 2차원 데이터셋 k-평균 산점도 비교
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments2, ax=axes[0])
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments5, ax=axes[1])
images.image.save_fig("6.11 blobs_Scatter_kmeans_compare")  
plt.show()


# k-평균 알고리즘이 실패하는 경우 
# 데이터셋의 클러스터 개수를 정확하게 알고 있더라고 항상 이를 구분해낼 수 있는 것은 아니다.
# 각 클러스터를 정의하는것이 중심 하나뿐이므로, 클러스터는 둥근 형태로 나타난다.

# 1. 클러스터간 밀도차이가 클 때 k-평균으로 찾은 클러스터 할당
#     k-평균은 클러스터에서 모든 방향이 똑같이 중요하다고 가정한다.
#     가운데 비교적 엉성한 영역에 비해 클러스터0,1은 중심에서 몰리 떨어진 포인트 들도 포함
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
print(X_varied, y_varied)
# 산점도
mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_varied)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1", "클래스 2"])
plt.title('클러스터간 밀도차이가 큰 2차원 데이터셋')
images.image.save_fig("6.11 blobs_Scatter_fail")  
plt.show()

y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('클러스터간 밀도차이가 큰 데이터셋 k-평균으로 찾은 클러스터 할당')
images.image.save_fig("6.11 blobs_Scatter_kmeans_fail")  
plt.show()

# 2. 원형이 아닌 클러스터를 구분하지 못하는 k-평균 알고리즘
X_spread, y_spread = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)
# 데이터가 길게 늘어지도록 변경한다.
transformation = rng.normal(size=(2, 2))
X_spread = np.dot(X_spread, transformation)

# 3개의 클러스터로 데이터 kmeans 알고리즘 적용
kmeans_spread = KMeans(n_clusters=3)
kmeans_spread.fit(X_spread)
y_spread_pred = kmeans_spread.predict(X_spread)

# 클러스터 할당과 클러스터 중심 나타내기
mglearn.discrete_scatter(X_spread[:, 0], X_spread[:, 1], kmeans_spread.labels_, markers='o')
mglearn.discrete_scatter(kmeans_spread.cluster_centers_[:, 0], kmeans_spread.cluster_centers_[:, 1], [0, 1, 2], 
    markers='^', markeredgewidth=2)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('길게 늘어진 데이터셋 k-평균으로 찾은 클러스터 할당')
images.image.save_fig("6.11 blobs_spread_Scatter_kmeans_fail")  
plt.show()

