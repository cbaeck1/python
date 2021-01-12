import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# [데이터셋의 스케일을 조정하거나 전처리하는 여러 방법]
# 두 개의 특성을 인위적으로 만든 이진 분류 데이터 셋이다.
# x축은 10~15의 값을 가지고  y축은 1~9의 값을 가진다.
# 오른쪽의 4가지 그림은 데이터를 기준이 되는 범위로 변환하는 네가지 방법이다
mglearn.plots.plot_scaling()
images.image.save_fig("10.scale_plot")     
plt.show()

# 1. StandardScaler 는 각 특성의 평균을 0 분산을 1로 변경하여 모든 특성이 같은 크기를 가지게 한다.
# 2. RobustScaler 는 특성들이 같은 스케일을 같게 된다는 통계적 측면에서는 비슷하지만, 중간값과 사분위값을 사용한다.
#    이런 방식 때문에 전체 데이터와 아주 동떨어진 데이터의 영향을 받지 않는다.(이런 데이터를 이상치라 한다, outlier)
# 3. MinmaxScaler 는 모든 특성이 정확하게 0과 1사이에 위치하도록 데이터를 변경한다.
# 4. Normalizer 는 벡터의 유클리디안 길이가 1의 되도록 데이터 포인트를 조정한다. 
#    (지름이 1인 구에 데이터 포인트를 투영한다.) 
#    이러한 정규하는 득성 벡터의 길이는 상관 없고 데이터의 방향이 중요할 때 많이 사용한다.

