import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 자주 사용하는 자료라 함수가 사이킷 런에 들어있기 때문에 이렇게 로드할 수 있다.
from sklearn.datasets import load_iris  

iris = load_iris()

dir(iris)
print(iris.feature_names)
print(iris.target_names)
print(iris.target, iris.target.shape)   
print(iris.target[iris.target==0].shape, iris.target[iris.target==1].shape, iris.target[iris.target==2].shape)
print(iris.data.shape, iris.data[:5])

# 훈련세트와 테스트세트로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

print(X_train[:5], y_train[:5])

iris_df = pd.DataFrame(X_train, columns=iris.feature_names)
pd.plotting.scatter_matrix(iris_df, c=y_train, s=60, alpha=0.8, figsize=[12,12])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
# n_neighbors 의 숫자가 커질수록 직선 경향이 커진다.
# 기본값은 5
model = KNeighborsClassifier(n_neighbors=5) 
# fit 데이터를 줄테니 모델을 만들어봐. > 이를 이용해 예측
model.fit(X_train, y_train) 

# 샘플이 하나라도 2차원 어레이를 넘겨야 한다
# predict 예측.
# score를 이용해 평가 : 실행할때마다 값은 바뀐다.
# 1 -> 0.9210526315789473
# 2 -> 0.9736842105263158
# 3 -> 0.9473684210526315
# 4 -> 0.9473684210526315
# 5 -> 0.9210526315789473

kind = model.predict([[6,3,4,1.5]]) 
print(kind)
score = model.score(X_test, y_test) 
print(score) #원래 잘 분리된 데이터라 높게 나왔다


pred_y=model.predict(X_test)
print(pred_y)
print(pred_y==y_test)
print((model.predict(X_test)==y_test).mean())

# Support Vector Machine 이 모듈의 특징은 아주 매끄러운 곡선을 그려준다. 
# 다차원에서는 매끄러운 곡면

from sklearn.svm import SVC
model = SVC(C=1.0, gamma=0.1)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)


