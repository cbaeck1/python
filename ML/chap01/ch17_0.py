import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import images.image

# 17. 3가지 합성된 신호 데이터셋
S = mglearn.datasets.make_signals()
print("S.shape: {}".format(S.shape))
print("S 타입: {}".format(type(S)))
print(S)


plt.figure(figsize=(6,2))
plt.plot(S,'-')
plt.xlabel("time")
plt.ylabel("signal")
plt.title("signals image")
images.image.save_fig("17. 3signals_image", "ml")
plt.show()

# 원본 데이터를 사용해 100개의 측정 데이터를 만든다.
A = np.random.RandomState(0).uniform(size=(100,3))
X = np.dot(S,A.T)
print("측정 데이터 형태 : {}".format(X.shape))


