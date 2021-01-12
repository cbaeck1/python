import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output

# 17. 3가지 합성된 신호 데이터셋
S = mglearn.datasets.make_signals()
print("S.shape: {}".format(S.shape))
print("S 타입: {}".format(type(S)))
print(S)

# 원본 데이터를 사용해 100개의 측정 데이터를 만든다.
A = np.random.RandomState(0).uniform(size=(100,3))
X = np.dot(S,A.T)
X = clean(X)
print("측정 데이터 형태 : {}".format(X.shape))

from sklearn.decomposition import NMF
nmf = NMF(n_components=3, random_state=42)
# Exception has occurred: ValueError array must not contain infs or NaNs
S_ = nmf.fit_transform(X)
print("복원한 신호(S_) 데이터 형태 : {}".format(S_.shape))

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)
models = [X, S, S_, H]
names = ['estimated 3', 'original', 'nmf', 'pca']
print("복원한 신호(H) 데이터 형태 : {}".format(H.shape))

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace':0.5}, subplot_kw={'xticks':(), 'yticks':()})
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:,:3],'-')
plt.title("signals image")
images.image.save_fig("10.17.3signals_image_pca_nmf")
plt.show()

