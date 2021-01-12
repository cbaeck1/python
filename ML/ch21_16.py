import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 16. 고유얼굴(eigenface) people
# 이미지는 RGB(적,녹,청)의 강도가 기록된 픽셀로 구성
# LWF 데이터셋에는 21명의 얼굴을 찍은 이미지가 총 1721개 있으며, 각 이미지의 크기는 87*65픽셀
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
print(people['DESCR']+ "\n...")
print("people.keys(): \n{}".format(people.keys()))
print("people 데이터의 형태: {}".format(people.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(people.target_names, np.bincount(people.target))}))
# print("특성 이름:\n{}".format(people.feature_names))
print(people.data, people.target)

image_shape = people.images[0].shape
print("people.images.shape : {}".format(people.images.shape))
print("클래스 개수 : {}".format(len(people.target_names)))
# 각 타깃이 나타난 횟수 
counts = np.bincount(people.target)
# 타깃별 이름과 횟수 출력
for i,(count,name) in enumerate(zip(counts,people.target_names)):
    print("{0:25} {1:3}".format(name,count))
    if (i+1) %3 ==0 :
        print()

# 데이터의 편중을 없애기위해 사람마다 50개의 이미지만 선택
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]]=1

X_people = people.data[mask]
y_people = people.target[mask]
# 0~255 사이의 흑백 이미지를 픽셀 값을 0~1 스케일로 조정
# MinMaxScaler 적용과 비슷
X_people = X_people/255.

# < 벡터 양자화 또는 분해 메서드로서의 k-평균 >
# k-평균이 군집 알고리즘이지만, k-평균과 PCA나 NMF같은 분해 알고리즘 사이에 유사점이 있다.
# PCA는 데이터에서 분산이 가장 큰 방향을 찾으려 하고,
# NMF는 데이터의 극단 또는 일부분에 상응되는 중첩할수 있는 성분을 찾는다.
# 두 방법 모두 데이터 포인트를 어떤 성분의 합으로 표현한다.
# 
# k-평균은 클러스터 중심으로 각 데이터 포인트를 표현한다.(즉, 하나의 성분으로 표현된다.)
# k-평균을 이렇게 각 포인트가 하나의 성분으로 분해되는 관점으로 보는것을 벡터 양자화라고 한다.

# 성분/클러스터중심 100개를 사용한 k-평균 / PCA / NMF의 이미지 재구성 비교 
# 테스트/훈련세트 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)

########################################################################
#  21. k-평균 군집
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
X_reconstructed_kmeams = kmeans.cluster_centers_[[kmeans.predict(X_test)]]

fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks':(), 'yticks':()})
fig.suptitle("추출한 성분")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")    
plt.title("people 추출한 성분")
images.image.save_fig("16.11 people_image_extract")
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(8, 8), subplot_kw={'xticks':(), 'yticks':()})
fig.suptitle("재구성")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeams, X_reconstructed_pca, X_reconstructed_nmf):
    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape), cmap='viridis')
    ax[3].imshow(rec_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")   
plt.title("people 재구성")
images.image.save_fig("16.11 people_image_restruct")
plt.show()


# k-평균을 사용한 벡터 양자화의 흥미로운 점은 입력 데이터의 차원보다 더 많은 클러스터를 사용해 인코딩
