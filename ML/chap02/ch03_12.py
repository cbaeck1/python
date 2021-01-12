import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 12. 뉴스그룹 데이터 : tuple
from sklearn.datasets import fetch_20newsgroups
newsdata = fetch_20newsgroups(subset='train')
print(newsdata['DESCR']+ "\n...")
print("newsdata.keys(): \n{}".format(newsdata.keys()))
print("newsdata.data의 type: {}".format(type(newsdata.data)))
print("newsdata.target의 type: {}".format(type(newsdata.target)))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(newsdata.target_names, np.bincount(newsdata.target))}))
print("특성 이름:\n{}".format(newsdata.filenames))
print(newsdata.data[:5], newsdata.target)
print(newsdata.data[0])

# 산점도를 그립니다. 2개의 특성과 1개의 타켓(2개의 값)
mglearn.discrete_scatter(newsdata.data[0], newsdata.data[1], newsdata.target)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.title("newsdata Scatter Plot")
images.image.save_fig("newsdata_Scatter")  
plt.show()

########################################################################
# 3. 나이브 베이즈 분류
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브 베이즈 모델
from sklearn.metrics import accuracy_score #정확도 계산

dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(newsdata.data)
print(X_train_dtm.shape)

# 
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
print(tfidfv.shape)

# 
mod = MultinomialNB()
mod.fit(tfidfv, newsdata.target)

clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) #테스트 데이터 갖고오기
X_test_dtm = dtmvector.transform(newsdata_test.data) #테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
print("정확도:", accuracy_score(newsdata_test.target, predicted)) #예측값과 실제값 비교

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

show_top10(clf, vectorizer, newsgroups_train.target_names)



