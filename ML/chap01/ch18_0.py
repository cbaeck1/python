import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 18. 숫자 데이터셋의 샘플 이미지 digits 
from sklearn.datasets import load_digits
digits = load_digits()
print(digits['DESCR']+ "\n...")
print("digits.keys(): \n{}".format(digits.keys()))
print("digits 데이터의 형태: {}".format(digits.data.shape))
print("클래스별 샘플 개수:\n{}".format(
      {n: v for n, v in zip(digits.target_names, np.bincount(digits.target))}))
print("특성 이름:\n{}".format(digits.feature_names))
print(digits.data, digits.target)

image_shape = digits.images[0].shape

print("digits.images.shape : {}".format(digits.images.shape))
print("클래스 개수 : {}".format(len(digits.target_names)))
# 각 타깃이 나타난 횟수 
counts = np.bincount(digits.target)
# 타깃별 이름과 횟수 출력
for i,(count, name) in enumerate(zip(counts, digits.target_names)):
    print("{0:25} {1:3}".format(name, count))
    if (i+1) %3 ==0 :
        print()

fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks':()})
for ax, image_shape in zip(axes.ravel(), digits.images):
    ax.imshow(image_shape)
plt.title("digits image")
images.image.save_fig("18. digits_image")
plt.show()


