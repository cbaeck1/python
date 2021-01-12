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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people,y_people,stratify=y_people, random_state=0)

########################################################################
# 1. k-최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("1-최근접 이웃 테스트 정확도 : {:.3f}".format(knn.score(X_test,y_test)))


