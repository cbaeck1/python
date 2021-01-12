# 파이썬 2와 3에 모두 호환되도록 필요한 모듈을 임포트
from __future__ import division, print_function, unicode_literals

import numpy as np

# 배열 생성
# 0 으로 채워진 배열
a = np.zeros(5)
print(a)

#x = np.array([[1, 2, 3], [4, 5, 6]])
#print("x:\n{}".format(x))