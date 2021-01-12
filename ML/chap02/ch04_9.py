import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 9. 두 개의 클래스를 가진 2차원 데이터셋 make_moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))
print("X 타입: {}".format(type(X)))
print("y 타입: {}".format(type(y)))
print(X[:5], y[:5])

########################################################################
# 4. 결정트리
from sklearn.tree import DecisionTreeClassifier
# max_depth = 1
tree1 = DecisionTreeClassifier(max_depth=1, random_state=0)
tree1.fit(X, y)
print("훈련 세트 깊이 1 정확도: {:.3f}".format(tree1.score(X, y)))
print("테스트 세트 깊이 1 정확도: {:.3f}".format(tree1.score(X, y)))

# export_graphviz 함수를 이용해 트리를 시각화
from sklearn.tree import export_graphviz
export_graphviz(tree1, out_file="./images/svg/9.4 make_moons_tree_depth1.dot", class_names=["Flase", "True"],
                feature_names=["특성1", "특성2"],
                impurity=False, filled=True)
with open("./images/svg/9.4 make_moons_tree_depth1.dot", encoding='utf8') as f:
    dot_graph = f.read()

# display(graphviz.Source(dot_graph))
# print(graphviz.Source(dot_graph))
image.save_graph_as_svg(dot_graph, "9.4 make_moons_decision_tree_depth1")  

# max_depth = 2
tree2 = DecisionTreeClassifier(max_depth=2, random_state=0)
tree2.fit(X, y)
print("훈련 세트 깊이 2 정확도: {:.3f}".format(tree2.score(X, y)))
print("테스트 세트 깊이 2 정확도: {:.3f}".format(tree2.score(X, y)))

export_graphviz(tree2, out_file="./images/svg/9.4 make_moons_tree_depth2.dot", class_names=["Flase", "True"],
                feature_names=["특성1", "특성2"],
                impurity=False, filled=True)
with open("./images/svg/9.4 make_moons_tree_depth2.dot", encoding='utf8') as f:
    dot_graph = f.read()
image.save_graph_as_svg(dot_graph, "9.4 make_moons_decision_tree_depth2")  

# max_depth = 3
tree3 = DecisionTreeClassifier(max_depth=3, random_state=0)
tree3.fit(X, y)
print("훈련 세트 깊이 3 정확도: {:.3f}".format(tree3.score(X, y)))
print("테스트 세트 깊이 3 정확도: {:.3f}".format(tree3.score(X, y)))

export_graphviz(tree3, out_file="./images/svg/9.4 make_moons_tree_depth3.dot", class_names=["Flase", "True"],
                feature_names=["특성1", "특성2"],
                impurity=False, filled=True)
with open("./images/svg/9.4 make_moons_tree_depth3.dot", encoding='utf8') as f:
    dot_graph = f.read()
image.save_graph_as_svg(dot_graph, "9.4 make_moons_decision_tree_depth3")  

# 각 분할된 영역이 (결정 트리의 리프) 한 개의 타깃값(하나의 클래스나 하나의 회귀 분석 결과)을 가질 때까지 반복
# 타깃 하나로만 이뤄진 리프 노드를 순수 노드 pure node
# 과대적합 모델 : 순수 노드로 이루어진 트리는 훈련 세트에 100% 정확하게 맞는다는 의미
# max_depth = 0
tree0 = DecisionTreeClassifier(random_state=0)
tree0.fit(X, y)
print("훈련 세트 리프노드 정확도: {:.3f}".format(tree0.score(X, y)))
print("테스트 세트 리프노드정확도: {:.3f}".format(tree0.score(X, y)))

export_graphviz(tree0, out_file="./images/svg/9.4 make_moons_tree_depth0.dot", class_names=["Flase", "True"],
                feature_names=["특성1", "특성2"],
                impurity=False, filled=True)
with open("./images/svg/9.4 make_moons_tree_depth0.dot", encoding='utf8') as f:
    dot_graph = f.read()
image.save_graph_as_svg(dot_graph, "9.4 make_moons_decision_tree_depth0")  


# max_depth를 12개 까지 
fig, axes = plt.subplots(3, 4, figsize=(20, 10))
# print(type(axes), type(axes.ravel()))
for i in range(12):
    # print(type(ax))
    ax = axes.ravel()[i]
    ax.set_title("트리 {}".format(i))
    tree = DecisionTreeClassifier(max_depth=i+1, random_state=0)
    tree.fit(X, y)
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

image.save_fig("9.4 make_moons_decision_tree_depth_stepbystep")  
plt.show()


# 과대적합을 막는 전략
# 1. 사전가지치기 : 트리 생성을 일찍 중단
#    1) 트리의 최대 깊이나 리프의 최대 개수를 제한 
#    2) 노드가 분할하기 위한 포인트의 최소 개수를 지정
# 2. 사후가지치기 : 트리를 만든 후 데이터 포인트가 적은 노드를 삭제하거나 병합
# scikit-learn은 사전 가지치기만 지원

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("X_train 타입: {}".format(type(X_train)))
print("y_train 타입: {}".format(type(y_train)))
print(X_train[:5], y_train[:5])

# 완전한 트리(모든 리프 노드가 순수 노드가 될 때까지 생성한 트리) 모델 생성 -> tree0
from sklearn.tree import DecisionTreeClassifier
tree0 = DecisionTreeClassifier(random_state=0)
tree0.fit(X_train, y_train)
print("훈련 세트 완전한 트리 정확도: {:.3f}".format(tree0.score(X_train, y_train)))
print("테스트 세트 완전한 트리 정확도: {:.3f}".format(tree0.score(X_test, y_test)))

# 1. 일정 깊이에 도달하면 트리의 성장을 멈춤
# max_depth=4 
tree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
tree4.fit(X_train, y_train)
print("훈련 세트 max_depth=4 정확도: {:.3f}".format(tree4.score(X_train, y_train)))
print("테스트 세트 max_depth=4 정확도: {:.3f}".format(tree4.score(X_test, y_test)))


# 트리의 특성 중요도 (feature importance)
# 각 특성에 대해 0은 전혀 사용되지 않았다는 뜻이고 1은 완벽하게 타깃 클래스를 예측했다는 의미
print("특성 중요도: {}".format(tree3.feature_importances_))

# 선형 모델의 계수를 시각화하는 것과 비슷한 방법으로 특성 중요도도 시각화
image.plot_feature_importances(tree3, X_train, list(range(2)))
plt.title('데이터로 학습시킨 결정 트리의 특성 중요도')
image.save_fig("9.4 make_moons_tree_depth3_feature_importances")  
plt.show()


# 특성과 클래스 사이에는 간단하지 않은 관계가 있음에 관한 예제
# X[1]에 있는 정보만 사용되었고 X[0]은 전혀 사용되지 않음
# 
tree_not_monotone = mglearn.plots.plot_tree_not_monotone()
plt.title('데이터로 학습한 결정 트리')
image.save_fig("9.4 make_moons_tree_depth3_not_monotone")  
plt.show()

# 장단점과 매개변수
# 10,000개의 샘플 정도면 SVM 모델이 잘 작동
# 100,000개 이상의 데이터셋에서는 속도와 메모리 관점에서 도전적인 과제
# 데이터 전처리와 매개변수 설정에 신경을 많이 써야 한다

# 커널 SVM에서 중요한 매개변수는 규제 매개변수 C





