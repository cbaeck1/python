import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

############################################################################
#  21. k-평균 군집
# k-평균(k-mean)군집은 가장 간단하고 널리 사용하는 군집 알고리즘이다.
# 데이터의 어떤 영역을 대표하는 클러스터 중심을 찾는다.
#   0. 클러스터 중심으로 삼을 데이터 포인트 3개를 무작위
#   1. 데이터 포인트를 가장 가까운 클러스터 중심에 할당한다.
#   2. 다음 클러스터에 할당된 데이터 포인트의 평균으로 클러스터 중심을 다시 지정한다.
#   이 두단계를 반복하고 할당되는 데이터 포인트에 변화가 없을때 종료한다.

mglearn.plots.plot_kmeans_algorithm()
images.image.save_fig("11.kmeans_algorithm_plot")     
plt.show()
# 삼각형 : 클러스터 중심
# 원 : 데이터 포인트

mglearn.plots.plot_kmeans_boundaries()
images.image.save_fig("11.kmeans_boundaries_plot")     
plt.show()

# < 벡터 양자화 또는 분해 메서드로서의 k-평균 >
# k-평균이 군집 알고리즘이지만, k-평균과 PCA나 NMF같은 분해 알고리즘 사이에 유사점이 있다.
# PCA는 데이터에서 분산이 가장 큰 방향을 찾으려 하고,
# NMF는 데이터의 극단 또는 일부분에 상응되는 중첩할수 있는 성분을 찾는다.
# 두 방법 모두 데이터 포인트를 어떤 성분의 합으로 표현한다.
# 
# k-평균은 클러스터 중심으로 각 데이터 포인트를 표현한다.(즉, 하나의 성분으로 표현된다.)
# k-평균을 이렇게 각 포인트가 하나의 성분으로 분해되는 관점으로 보는것을 벡터 양자화라고 한다.

# 장점
# k-평균은 비교적 이해하기 쉽고 구현도 쉬울 뿐만 아니라 비교적 빠르기 때문에 가장 인기있는 군집 알고리즘이다.
# k-평균은 대용량 데이터셋에도 잘 작동하지만, MinBatchKMeans도 제공한다.
# 단점
#   무작위 초기화를 사용하여 알고리즘의 출력이 난수 초깃값에 따라 달라진다.
#   클러스터의 모양을 가정하고 있어서 활용범위가 비교적 제한적이다.

#  22. 병합 군집 ( agglomerative clustering )
#    병합군집 알고리즘은 사작할 때 각 포인트를 하나의 클러스터로 지정하고, 
#    그 다음 어떤 종료 조건을 만족할 때까지 가장 비슷한 두 클러스터를 합쳐나간다.

mglearn.plots.plot_agglomerative_algorithm()
images.image.save_fig("12.agglomerative_algorithm_plot")     
plt.show()

#  ward : 기본값인 ward 연결은 모든 클러스터 내의 분산을 가장 작게 증가시키는 두 클러스터를 합친다. 
#         -> 크기가 비교적 비슷한 클러스터가 만들어진다.
#  average : 클러스터 포인트 사이의 평균 거리가 가장 짧은 두 클러스터를 합친다.

#  초기에 각 포인트가 하나의 클러스터이다.
#  그 다음 각 단계에서 가장 가까운 두 클러스터가 합쳐진다.
#  알고리즘 작동 특성상 병합 군집은 새로운 데이터 포인트에 대해서는 예측을 할 수 없다.
#  그러므로 병합 군집은 predict 메서드가 없다.
#  대신 훈련 세트로 모델을 만들고 클러스터 소속 정보를 얻기 위해서 fit_predict 메서드를 사용한다.

#  23. DBSCAN : density-based spatialclustering of applications with noise
#    장점 : 클러스터의 개수를 미리 지정할 필요가 없다.
#    이 알고리즘은 복잡한 형상도 찾을 수 있으며, 어떤 클래스에도 속하지 않는 포인트를 구분할 수 있다.
#    병합군집이나 k-평균보다는 다소 느리지만 비교적 큰 데이터셋에도 적용할 수 있다.
#    DBSCAN은 특성 공간에서 가까이 있는 데이터가 많아 붐비는 지역의 포인트를 찾는다.
#    이런 지역을 특성 공간의 밀집 지역(dense region)이라고 한다.
#    DBSCAN의 아이디어는 데이터의 밀집 지역이 한 클러스터를 구성하며 비교적 비어있는 지역을 경계로 다른 클러스터와 구분된다는 것이다.
#    밀집 지역에 있는 포인트를 핵심샘플(핵심포인트)라고 한다.
#    DBSCAN에는 두 개의 매개변수 min_samples 와 eps 가 있다.
#    한 데이터 포인트에서 eps 거리 안에 데이터가 min_samples 개수만큼 있으면 이 데이터 포인트를 핵심샘플로 분류한다.
#    eps 보다 가까운 핵심샘플은 DBSCAN 에 의해 동일한 클러스터로 합쳐진다.
#    이 알고리즘은 시작할 때 무작위로 포인트를 선택한다. 그런 다음 그 포인트에서 eps 거리 안의 모든 포인트를 찾는다.
#    만약 eps 거리 안에 있는 포인트 수가 min_samples 보다 적다면 그 포인트는 어떤 클래스에도 속하지 않는 잡음(noise)으로 레이블한다.
#    eps 거리 안에 min_samples 보다 많은 포인트가 있다면 그 포인트는 핵심샘플로 레이블하고 새로운 클러스터 레이블을 할당한다.
#    그런 다음 그 포인트의 모든 이웃을 살핀다.
#    이런 식으로 계속 진행하려 글러스터는 eps 거리 안에 더 이상 핵심샘플이 없을때까지 자라나며
#    그런 다음 아직 방문하지 못한 포인트를 선택하여 같은 과정을 반복한다.

mglearn.plots.plot_dbscan()
images.image.save_fig("13.DBSCAN_plot")     
plt.show()
