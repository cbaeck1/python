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
#  22. 병합 군집 ( agglomerative clustering )
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["cluster 0", "cluster 1"], loc="best")
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.title('임의의 데이터셋에 병합 군집을 사용한 세 개의 클러스터 할당')
images.image.save_fig("6.12 blobs_agglomerative_scatter")  
plt.show()


# <계층적 군집과 덴드로그램>
# 병합 군집은 계층적 군집을 만든다. 
# 군집이 반복하여 진행되면서 모든 포인트는 하나의 포인트를 가진 클러스터에서 마지막 클러스터까지 이동하게 된다.
# 다음은 각 클러스터가 더 작은 클러스터로 어떻게 나뉘는지 잘 보여준다.
# 병합군집으로 생성한 계층적 군집

mglearn.plots.plot_agglomerative()
plt.title('병합군집으로 생성한 계층적 군집(2차원 데이터)')
images.image.save_fig("6.12 agglomerative_scatter")  
plt.show()

# 계층 군집을 시각화하는 또 다른 도구인 덴드로그램(dendrogram)은 다차원 데이터셋을 처리
# scikit-learn은 아직 덴드로그램을 그리는 기능은 제공하지 않는다
# SciPy를 사용해 손쉽게 만들 수 있다.

X_dendrogram, y_dendrogram = make_blobs(random_state=0, n_samples=12)
# 데이터 배열 X_dendrogram 에 ward 함수를 적용
# SciPy 의 ward 함수는 병합 군집을 수행할 때 생성된 거리 정보가 담긴 배열을 반환한다.
from scipy.cluster.hierarchy import dendrogram, ward
linkage_array = ward(X_dendrogram)
# 클러스터 간의거리 정보가 담긴 linkage_array를 사용해 덴드로그램을 그린다.
dendrogram(linkage_array)

# 두개와 세개의 클러스터를 구분하느 커트라인
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, '2 cluster', va='center', fontdict={'size':15})
ax.text(bounds[1], 4, '3 cluster', va='center', fontdict={'size':15})
plt.xlabel("sample num")
plt.ylabel("cluster dist")
plt.title('병합군집으로 생성한 계층적 군집(다차원 데이터)')
images.image.save_fig("6.12 agglomerative_dendrogram")  
plt.show()

# y 축은 병합알고리즘에서 두 클러스터가 합쳐지는 것을
# 가지의 길이는 합쳐진 클러스터가 얼마나 멀리 떨어져 있는지 보여준다

# 하지만 병합 군집은 two_moons데이터셋과 같은 복잡한 형상을 구분하지 못한다.



