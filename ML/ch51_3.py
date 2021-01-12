import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 3. 위스콘신 유방암 Wisconsin Breast Cancer 데이터셋입니다(줄여서 cancer라고 하겠습니다). 
# 각 종양은 양성benign(해롭지 않은 종양)과 악성malignant(암 종양)으로 레이블되어 있고, 
# 조직 데이터를 기반으로 종양이 악성인지를 예측할 수 있도록 학습하는 것이 과제
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer['DESCR']+ "\n...")
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))
print(cancer.data, cancer.target)
print(cancer.data[:,:2])

################################################################################
# 51. 주성분 분석(PCA) - 유방암 데이터셋 시각화 하기 
# 주성분 분석은 특성들이 통계적으로 상관관계가 없도록 데이터셋을 회전시키는 기술이다.
# 회전한 뒤에 데이터를 설명하는데 얼마나 중요하냐에 따라 특성 일부만 선택된다.
# 인위적으로 만든 2차원 데이터셋을 사용하여 PCA효과를 나타낸 그래프
mglearn.plots.plot_pca_illustration()
images.image.save_fig("51.3.breast_cancer_Scatter")  
plt.show()

# 1. PCA 알고리즘은 성분1이라고 쓰여있는 분산이 가장 큰 방향을 찾는다.
#   이 방향(벡터)는 이 성분1에 대한 가장 많은 정보를 담고 있는 방향이다.
# 2. 그 다음 첫번째 방향과 직각인 방향중에 가장 많은 정보를 담은 방향을 찾는다. (component 2)
#   이런 과정을 거쳐 찾은 방향을 데이터에 있는 주된 분산의 방향이라고 하여 주성분(principal component) 이라고 한다. 
#   일반적으로 원본 특성의 개수 만큼 있다.
# 이 변환은 데이터에서 노이즈를 제거하거나 주성분에서 유지되는 정보를 시각화하는데 종종 사용

# 그림1: 원본 데이터 포인트를 색으로 구분
# 그림2 : 주성분 1과 2를 각각 x출과 y축에 나란하게 회전
# 그림3 : 주성분 일부만 남기는 차원 축소로 첫번째 주성분만 유지
# 그림4 : 그림 3에서 데이터에 평균을 더해 반대로 회전

# 유방암 데이터의 클래스별 특성 히스토그램
# 초록색 : 양성, 파란색: 음성 클래스
'''
fig,axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]
ax = axes.ravel()
for i in range(30):
    _,bins = np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i],bins=bins, color=mglearn.cm3(0),alpha=.5)
    ax[i].hist(benign[:,i],bins=bins, color=mglearn.cm3(2),alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel("attr size")
ax[0].set_ylabel("frequency")
ax[0].legend(["neg","pos"],loc="best")
fig.tight_layout()
images.image.save_fig("51.3.breast_cancer_histogram")  
plt.show()
'''
# 예) worst concave points 특성은 두 히스토그램이 확실히 구분되어 매우 유용한 특성이지만 특성간의 상호작용에 대해선 전혀 알려주지 못한다


# 훈련 세트, 테스트 세트
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

# 처음 두개의 주성분을 사용해 그린 유방암 데이터셋의 2차원 산점도
# pca 적용 전에 StandardScaler 를 사용해 데이터 스케일 조정 : 각 특성의 분산이 1이 되도록 스케일 조정
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(cancer.data)
X_scaled = standard_scaler.transform(cancer.data)

#standard_scaler.fit(X_train)
#X_train_scaled_standard = standard_scaler.transform(X_train)
#X_test_scaled_standard = standard_scaler.transform(X_test)

standard_scaler.fit(cancer.data)
X_train_scaled_standard = standard_scaler.transform(cancer.data)

# PCA 객체를 생성하고 fit메서드를 호출해 주성분을 찾고, transform 메서드를 호출해 데이터를 회전시키고 차원을 축소한다.
# 기본값일때 PCA는 데이터를 회전만 시키고 모든 주성분을 유지한다.
# 데이터의 차원을 줄이려면 PCA 객체를 지정하면 된다.
from sklearn.decomposition import PCA
# pca = PCA(n_components=2) --> ValueError: array must not contain infs or NaNs
# 원인 : The numpy array shape is (x, y), dtype is float64. 
#        The minimum value in the array is 0.0, the maximum value is 1.7976931348623157e+308. 
#        The array does not contain infs or NaNs but I get an error
# 처리 : whiten=True and the largest value equal to the largest possible value of a float64

# LinAlgError: SVD did not converge
# 처리1 : dataFrame : X_train_scaled_standard.dropna(inplace=True)
#         numpy : X[~np.isnan(X).any(axis=1)]
# 데이터 첫 2개의 성분만 유지한다.
pca = PCA(n_components=2)
# PCA 모델 만들기
pca.fit(X_scaled)
# 처음 두개의 주성분을 사용해 데이터 변환
X_pca = pca.transform(X_scaled)
print("원본 데이터 형태 : {}".format(str(X_scaled.shape)))
print("축소된 데이터 형태 : {}".format(str(X_pca.shape)))


print(X_train_scaled_standard.shape)
print(X_train_scaled_standard)
X_train_scaled_standard = X_train_scaled_standard[~np.isnan(X_train_scaled_standard)]
print(X_train_scaled_standard.shape)
print(X_train_scaled_standard)
X_train_scaled_standard = X_train_scaled_standard.reshape(-1,1)
print(X_train_scaled_standard.shape)

# 데이터 첫 2개의 성분만 유지한다.
pca = PCA(n_components=2, whiten=True, svd_solver='full')
# PCA 모델 만들기
pca.fit(X_train_scaled_standard)
# 처음 두개의 주성분을 사용해 데이터 변환
X_pca = pca.transform(X_train_scaled_standard)
print("원본 데이터 형태 : {}".format(str(X_train_scaled_standard.shape)))
print("축소된 데이터 형태 : {}".format(str(X_pca.shape)))
print("PCA 주성분: {}".format(pca.components_.shape))
print("PCA 주성분: {}".format(pca.components_))

# 두개의 주성분을 그래프로 나타내자.
# PCA의 단점은 그래프의 두 축을 해석하기 쉽지 않다
plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
plt.legend(["neg","pos"],loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("1st attr")
plt.ylabel("2nd attr")
plt.title("유방암 데이터의 PCA 주성분 특성 산점도")
images.image.save_fig("3.11.breast_cancer_pca_scatter")  
plt.show()


# 히트맵 시각화 하기
# 첫번째 주성분 (첫번째 가로 줄)의 모든특성은 부호가 같다. 모든 특성 사이에 공통의 상호관계가 있다. 한 특성의 값이 커지면 다른 값들도 높아진다
# 두번째 주성분의 특성은 부호가 섞여있다. 따라서 2번째 주성분의 축이 가지는 의미는 파악하기 힘들다.
plt.matshow(pca.components_, cmap="viridis")
plt.yticks([0, 1], ["comp 1", "comp 2"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("attr")
plt.ylabel("principle comp")
plt.title("유방암 데이터의 PCA 주성분 HitMap")
images.image.save_fig("3.11.breast_cancer_pca_hitmap")  
plt.show()


