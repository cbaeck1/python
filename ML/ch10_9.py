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

###############################################################################
# 1. 타깃값으로 군집 평가하기 : 군집 알고리즘의 결과를 실제 정답 클러스터와 비교하여 평가할 수 있는 지표
#  1. ARI (adjusted rand index)
#     ARI : 1(최적일 때)와 0(무작위로 분류될 때)
#  2. NMI (normalized mutual information)
# 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks':()})
# 3가지 알고리즘들 리스트
algos = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
# 무작위로 할당한 클러스터
from sklearn.metrics.cluster import adjusted_rand_score
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
axes[0].set_title("random assign - ARI : {:.2f}".format(adjusted_rand_score(y, random_clusters)))
for ax, algo in zip(axes[1:], algos):
    clusters = algo.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
    ax.set_title("{} - ARI: {:.2f}".format(algo.__class__.__name__, adjusted_rand_score(y, clusters)))

# plt.title('복잡한 모양의 클러스터 군집 알고리즘 비교')
images.image.save_fig("10.9.moons_spiral_scatter_adjusted_rand_score")  
plt.show()

# 2. 타깃값 없이 군집 평가하기 - 실루엣 계수
#    군집 알고리즘을 적용할 때 보통 그 결과와 비교할 타깃값이 없다.
#    타깃값이 필요 없는 군집용 지표로는 실루엣 계수 (silhouette coefficient)가 있다.
#    그러나 이 지표는 실제로 잘 동작하진 않는다.
#    실루엣 점수는 클러스터의 밀집 정도를 계산하는 것으로, 높을수록 좋으며, 최대 점수는 1이다.

# 실루엣 계수 사용하여 k-평균, 병합군집, DBSCAN 알고리즘을 비교
fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks':()})
# 3가지 알고리즘들 리스트
# algos = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
# random_state = np.random.RandomState(seed=0)
# random_clusters = random_state.randint(low=0, high=2, size=len(X))
# 무작위로 할당한 클러스터
from sklearn.metrics.cluster import silhouette_score
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
axes[0].set_title("random assign : {:.2f}".format(silhouette_score(X_scaled, random_clusters)))
for ax, algo in zip(axes[1:], algos):
    clusters = algo.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
    ax.set_title("{} : {:.2f}".format(algo.__class__.__name__, silhouette_score(X_scaled, clusters)))
# plt.title('복잡한 모양의 클러스터 군집 알고리즘 비교')
images.image.save_fig("10.9.moons_spiral_scatter_silhouette_score")  
plt.show()





