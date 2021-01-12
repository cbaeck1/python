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
X, y = make_blobs(random_state=42)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X, y)

# 산점도
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1", "클래스 2"])
plt.title('세 개의 클래스를 가진 2차원 데이터셋')
images.image.save_fig("6. blobs_Scatter")  
plt.show()

# X 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 range로 표현
# 사용할 특성의 갯수을 설정
blobs_df = pd.DataFrame(X, columns=range(2))
# 데이터프레임을 사용해  특성별 Historgram
blobs_df.plot.hist(alpha=0.5, bins=30)
plt.title("blobs_df Histogram Plot")
images.image.save_fig("6. blobs_df_Histogram")
plt.show() 


