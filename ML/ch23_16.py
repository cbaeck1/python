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

########################################################################
# 얼굴 데이터셋으로 군집 알고리즘 비교

from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

# 1. DBSCAN으로 얼굴 데이터셋 분석하기
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("고유한 레이블 \n{}".format(np.unique(labels)))
# 레이블이 -1일 뿐이므로 모든 데이터가 DBSCAN의 잡음 포인트

# 바꿀 수 있는 매개변수는 eps, min_samples 두가지
# eps 값을 크게하여 각 포인트의 이웃을 늘리거나, min_samples 값을 낮추어 클러스터에 모들 포인트 수를 줄인다
# min_samples=3 인 경우
dbscan3 = DBSCAN(min_samples=3)
labels3 = dbscan3.fit_predict(X_pca)
print("고유한 레이블 min_samples=3 \n{}".format(np.unique(labels3)))
# min_samples=3 로 줄여도 모두 잡음포인트로 레이블 되었다.

# min_samples=3, eps=15 인 경우
dbscan15 = DBSCAN(min_samples=3, eps=15)
labels15 = dbscan15.fit_predict(X_pca)
print("고유한 레이블 min_samples=3, eps=15\n{}".format(np.unique(labels15)))

# eps를 15로 크게 늘렸더니 클러스터 하나와 잡음 포인트를 얻었다.
# 잡음 포인트와 클러스터에 속한 포인트 수 세기, bincount는 음수를 받을수 없어 +1을 한다.
print("클러스터 별 포인트 수: {}".format(np.bincount(labels15 + 1)))

# 잡음 포인트는 총 19개이므로 모두 시각화 하여 확인해보자
noise = X_people[labels15 == -1]
fig, axes = plt.subplots(3, 9, subplot_kw={'xticks':(), 'yticks':()}, figsize=(12, 4))
for img, ax in zip(noise, axes.ravel()):
    ax.imshow(img.reshape(image_shape), vmin=0, vmax=1)
# plt.title("people 잡음 포인트로 잡힌 이미지 확인하기")
images.image.save_fig("16.11 people_image_min_samples_eps")
plt.show()

# 이렇게 특이한 것을 찾아내는 이런 종류의 분석을 이상치 검출이라고 한다.
# 위에서는 eps=15 로 하여 하나의 클러스터를 얻었다.
# 더 많은 클러스터를 찾으려면 eps 를 0.5~15 사이 정도로 줄여야 한다.

# eps값에 따른 클러스터의 변화
# 잡음 포인트와 클러스터에 속한 포인트 수 세기
# bincount는 음수를 받을수 없어 +1을 한다.
for eps in [1, 3, 5, 7, 9, 11, 13]:
    print("\neps={}".format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)    
    print("클러스터 수 {}".format(len(np.unique(labels))))
    print("클러스터 크기 {}".format(np.bincount(labels + 1)))

# eps=7 에서 DBSCAN으로 찾은 클러스터
dbscan7 = DBSCAN(min_samples=3, eps=7)
labels7 = dbscan7.fit_predict(X_pca)
for cluster in range(max(labels7) + 1):
    mask = labels7 == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks':(), 'yticks':()})  
    for img, label, ax in zip(X_people[mask], y_people[mask], axes):
        ax.imshow(img.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1])

    images.image.save_fig(people.target_names[label].split()[-1])
    plt.show()


# 2. k-평균으로 얼굴 데이터셋 분석하기
#   병합군집과 k-평균은 비슷한 크기의 클러스터
from sklearn.cluster import KMeans
# k-평균으로 클러스터 추출
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print("k-평균의 클러스터 크기 : {}".format(np.bincount(labels_km)))

fig, axes = plt.subplots(2, 5, figsize=(12, 4), subplot_kw={'xticks':(), 'yticks':()})  
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
images.image.save_fig("16.11 people_image_kmeans")
plt.show()

mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
images.image.save_fig("16.11 people_image_kmeans_faces")
plt.show()


