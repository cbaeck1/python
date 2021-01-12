import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 10. 두 개의 클래스를 가진 2차원 데이터셋
from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print(X, y)

# 산점도
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
plt.legend(["클래스 0", "클래스 1"])
plt.title('두 개의 클래스를 가진 2차원 데이터셋')
images.image.save_fig("10.make_handcrafted_Scatter")  
plt.show()


# X 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 range로 표현
# 사용할 특성의 갯수을 설정
handcrafted_df = pd.DataFrame(X, columns=range(2))
# 데이터프레임을 사용해  특성별 Historgram
handcrafted_df.plot.hist(alpha=0.5, bins=30)
plt.title("handcrafted_df Histogram Plot")
images.image.save_fig("10.make_handcrafted_Histogram")
plt.show() 




