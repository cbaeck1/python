import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 18. 숫자 데이터셋의 샘플 이미지 digits 
from sklearn.datasets import load_digits
digits = load_digits()
print(digits['DESCR']+ "\n...")
print("digits.keys(): \n{}".format(digits.keys()))
print("digits 데이터의 형태: {}".format(digits.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(digits.target_names, np.bincount(digits.target))}))
print("특성 이름:\n{}".format(digits.feature_names))
print(digits.data, digits.target)

image_shape = digits.images[0].shape

print("digits.images.shape : {}".format(digits.images.shape))
print("클래스 개수 : {}".format(len(digits.target_names)))
# 각 타깃이 나타난 횟수 
counts = np.bincount(digits.target)
# 타깃별 이름과 횟수 출력
for i,(count, name) in enumerate(zip(counts, digits.target_names)):
    print("{0:25} {1:3}".format(name, count))
    if (i+1) %3 ==0 :
        print()

#########################################################################
# 51. 주성분 분석 (principal component analysis)
# 처음 두 개의 주성분을 사용한 숫자 데이터셋의 산점도
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(digits.data)

# 처음 2개의 주성분으로 데이터 변환
digits_pca = pca.transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color=colors[digits.target[i]], 
        fontdict={'weight':'bold', 'size':9})
plt.xlabel("1st comp")
plt.ylabel("2nd comp")
plt.title("두 개의 주성분")
images.image.save_fig("18.31 digits_image_pca")
plt.show()

# 33. t-SNE 알고리즘 (t-distributed stochastic neighbor embedding)
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
digits = load_digits()
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
# colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color=colors[digits.target[i]], fontdict={'weight':'bold', 'size':9})
plt.xlabel("tsne 1")
plt.ylabel("tsne 2")
plt.title("t-SNE를 적용")
images.image.save_fig("18.33 digits_image_tsne")
plt.show()




