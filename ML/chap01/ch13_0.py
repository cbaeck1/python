import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image


# 13. 동물트리
mglearn.plots.plot_animal_tree()

plt.title('몇 가지 동물들을 구분하기 위한 결정 트리')
images.image.save_fig("13.animal_tree")  
plt.show()
