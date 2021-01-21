import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

from IPython.display import display 
import graphviz

# 20. life

# 1.1 판다스로 데이터프레임 만들기
dataframe = pd.read_csv('d:/data/trans서울중구.csv')
print(dataframe.head(), dataframe.shape)

# Categorical
X_dataframe = dataframe.loc[:,'매출_금액':'industry']
X_dataframe.industry = pd.Categorical(X_dataframe.industry)
X_dataframe.industry = X_dataframe.industry.cat.codes
X_dataframe = X_dataframe.apply(pd.to_numeric)
print(X_dataframe.head(), X_dataframe.shape, X_dataframe.info())

# numpy array
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X_dataframe)
X_life = imputer.transform(X_dataframe)

Y_life = dataframe['TARGET']
print(Y_life.head(), Y_life.shape)

# 1.2 데이터프레임을 훈련 세트, 검증 세트, 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_life, Y_life, stratify=Y_life, random_state=66)
print("X_train 크기: {}{}".format(X_train.shape, X_train.dtype))
print("y_train 크기: {}{}".format(y_train.shape, y_train.dtype))
print("X_test 크기: {}{}".format(X_test.shape, X_test.dtype))
print("y_test 크기: {}{}".format(y_test.shape, y_test.dtype))

########################################################################
# 4. 결정트리
# 유방암 데이터셋을 이용하여 사전 가지치기의 효과를 확인
# 완전한 트리(모든 리프 노드가 순수 노드가 될 때까지 생성한 트리) 모델 생성 -> tree0
from sklearn.tree import DecisionTreeClassifier
tree0 = DecisionTreeClassifier(random_state=0)
tree0.fit(X_train, y_train)
print("훈련 세트 완전한 트리 정확도: {:.3f}".format(tree0.score(X_train, y_train)))
print("테스트 세트 완전한 트리 정확도: {:.3f}".format(tree0.score(X_test, y_test)))

# export_graphviz 함수를 이용해 트리를 시각화
from sklearn.tree import export_graphviz
export_graphviz(tree0, out_file="ML/images/svg/20.life_tree.dot", class_names=["악성", "양성"],
                feature_names=X_dataframe.columns,
                impurity=False, filled=True)
with open("ML/images/svg/20.life_tree.dot", encoding='utf8') as f:
    dot_graph = f.read()

# display(graphviz.Source(dot_graph))
# print(graphviz.Source(dot_graph))
images.image.save_graph_as_svg(dot_graph, "4.20.life_decision_tree")  


# 과대적합을 막는 전략
# 1. 사전가지치기 : 트리 생성을 일찍 중단
#    1) 트리의 최대 깊이나 리프의 최대 개수를 제한 
#    2) 노드가 분할하기 위한 포인트의 최소 개수를 지정
# 2. 사후가지치기 : 트리를 만든 후 데이터 포인트가 적은 노드를 삭제하거나 병합
# scikit-learn은 사전 가지치기만 지원

# 1. 일정 깊이에 도달하면 트리의 성장을 멈춤
# max_depth=4 
tree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
tree4.fit(X_train, y_train)
print("훈련 세트 max_depth=4 정확도: {:.3f}".format(tree4.score(X_train, y_train)))
print("테스트 세트 max_depth=4 정확도: {:.3f}".format(tree4.score(X_test, y_test)))

export_graphviz(tree4, out_file="ML/images/svg/20.life_tree_depth4.dot", class_names=["악성", "양성"],
                feature_names=X_dataframe.columns,
                impurity=False, filled=True)
with open("ML/images/svg/20.life_tree_depth4.dot", encoding='utf8') as f:
    dot_graph = f.read()

# 유방암 데이터셋으로 만든 결정 트리
#   1) 경계값
#   2) samples 는 각 노드에 있는 샘플의 수
#   3) value 는 클래스당 샘플의 수
images.image.save_graph_as_svg(dot_graph, "4.20.life_decision_tree_depth4")  

# 트리의 특성 중요도 (feature importance)
# 각 특성에 대해 0 은 전혀 사용되지 않았다는 뜻이고 1 은 완벽하게 타깃 클래스를 예측했다는 의미
# 특성 중요도의 전체 합은 1 
# feature_importance_ 값이 낮다고 해서 이 특성이 유용하지 않다는 뜻은 아니고 단지 트리가 그 특성을 선택하지 않았을 뿐
print("특성 중요도:\n{}".format(tree4.feature_importances_))

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("특성 중요도")
#     plt.ylabel("특성")
#     plt.ylim(-1, n_features)

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
# 첫 번째 노드에서 사용한 특성(“worst radius”)이 가장 중요한 특성
images.image.plot_feature_importances(tree4, X_dataframe, X_dataframe.columns)
plt.title('life 데이터로 학습시킨 결정 트리의 특성 중요도')
images.image.save_fig("20.life_tree_depth4_feature_importances")  
plt.show()

# 특성과 클래스 사이에는 간단하지 않은 관계가 있음에 관한 예제
# X[1]에 있는 정보만 사용되었고 X[0]은 전혀 사용되지 않음
# X[1] 값이 높으면 클래스 0 이고 값이 낮으면 1 이라고(또는 그 반대로) 말할 수 없습니다
tree_not_monotone = mglearn.plots.plot_tree_not_monotone()
# print(type(tree_not_monotone))
# tree_not_monotone.render('cancer_decision_tree_not_monotone',view=True)
images.image.save_graph_as_svg(tree_not_monotone, "4.20.life_decision_tree_not_monotone")  

# 
plt.title('X[1]에 있는 정보만 사용 학습한 결정 트리')
images.image.save_fig("4.20.life_decision_tree_not_monotone")  
plt.show()


# DecisionTreeRegressor로 구현된 회귀 결정 트리에서도 비슷하게 적용
# 외삽 (extrapolation : 훈련 데이터의 범위 밖의 포인트) 에 대해 예측을 할 수 없슴.

