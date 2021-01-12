import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 7. scikit-learn에 구현된 나이브 베이즈 분류기는 
#    GaussianNB, BernoulliNB, MultinomialNB 
# BernoulliNB 분류기는 각 클래스의 특성 중 0이 아닌 것이 몇 개인지 셉니다
# MultinomialNB 분류기는 클래스별로 특성의 평균을 계산
# GaussianNB 분류기는 클래스별로 각 특성의 표준편차와 평균을 저장
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

# 0 [0, 1, 0, 1] [0, 0, 0, 1] : 첫번째, 세번째 데이터 포인트
# 1 [1, 0, 1, 1] [1, 0, 1, 0] : 두번째, 네번째 데이터 포인트
counts = {}
for label in np.unique(y):
    print(label, X[y == label],  X[y == label].sum(axis=0))
    # 클래스마다 반복
    # 특성마다 1이 나타난 횟수를 센다.
    counts[label] = X[y == label].sum(axis=0)
print("BernoulliNB(특성 카운트):{}".format(counts))
print("MultinomialNB(특성 평균):{}".format(counts[1].mean()))
print("GaussianNB(특성 표준편차):{}".format(counts[1].std()))



########################################################################
# 3. 
