import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image
# 12. 뉴스그룹 데이터
from sklearn.datasets import fetch_20newsgroups
newsdata = fetch_20newsgroups(subset='train')
print(newsdata['DESCR']+ "\n...")
print("newsdata.keys(): \n{}".format(newsdata.keys()))
print("뉴스그룹 데이터의 형태: {}".format(newsdata.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(newsdata.target_names, np.bincount(newsdata.target))}))
print("특성 이름:\n{}".format(newsdata.feature_names))
print(newsdata.data, newsdata.target)
print(newsdata.data[:,:2])



