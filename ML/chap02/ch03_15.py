import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 15. load_wine
from sklearn import datasets
wine = datasets.load_wine()

# print the names of the 13 features
print(wine['DESCR']+ "\n...")
print("wine.keys(): \n{}".format(wine.keys()))
print("데이터의 형태: {}".format(wine.data.shape))
# print the label type of wine(class_0, class_1, class_2)
print("특성 이름:\n{}".format(wine.feature_names))
print("Labels: ", wine.target_names)
print(wine.data, wine.target)
print(wine.target.mean(), wine.target.min(), np.percentile(wine.target,25), np.percentile(wine.target,75), wine.target.max())
print(wine.data[0:5], wine.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

########################################################################
# 3. 나이브 베이즈 분류
# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# 장점
# 1. 간단하고, 빠르며, 정확한 모델입니다.
# 2. computation cost가 작습니다. (따라서 빠릅니다.)
# 3. 큰 데이터셋에 적합합니다.
# 4. 연속형보다 이산형 데이터에서 성능이 좋습니다.
# 5. Multiple class 예측을 위해서도 사용할 수 있습니다.
# 단점
# feature 간의 독립성이 있어야 합니다. 
# 하지만 실제 데이터에서 모든 feature가 독립인 경우는 드뭅니다. 
# 장점이 많지만 feature가 서로 독립이어야 한다는 크리티컬한 단점이 있습니다.

