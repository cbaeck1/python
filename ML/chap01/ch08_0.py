import pandas as pd
import numpy as np
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import images.image

# 8. 메모리 가격 동향 데이터 셋 ram_prices DataFrame
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
print("데이터의 형태: {}".format(ram_prices.shape))
print("ram_prices 타입: {}".format(type(ram_prices)))
print(ram_prices)

# 
plt.plot(ram_prices.date, ram_prices.price)
plt.ylim(0, 100000000)
plt.xlabel("년")
plt.ylabel("가격 ($/Mbyte_")
plt.title("램 가격 동향")
images.image.save_fig("8.ram_prices_plot_by_raw")  
plt.show()

# 가격을 로그 스케일로 바꾸었기 때문에 비교적 선형적인 관계
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("년")
plt.ylabel("가격 ($/Mbyte_")
plt.title("로그 스케일로 그린 램 가격 동향")
images.image.save_fig("8.ram_prices_plot")  
plt.show()

# 2000년 이전을 훈련 데이터로, 2000년 이후를 테스트 데이터로 만듭니다.
data_train = ram_prices[ram_prices.date < 2000] 
data_test = ram_prices[ram_prices.date >= 2000]
# 가격 예측을 위해 날짜 특성만을 이용합니다.
X_train = data_train.date[:, np.newaxis]
# 데이터와 타깃의 관계를 간단하게 만들기 위해 로그 스케일로 바꿉니다.
y_train = np.log(data_train.price)
#
X_test = data_test.date[:, np.newaxis]
y_test = np.log(data_test.price)

print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("X_train 타입: {}".format(type(X_train)))
print("y_train 타입: {}".format(type(y_train)))
print(X_train[:5], y_train[:5])


# 비교 1:전체 2:X_train 3:X_test
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for X, y, title, ax in zip([ram_prices.date, X_train, X_test], [ np.log(ram_prices.price), y_train, y_test], 
    ['전체','X_train','X_test'], axes):
  # 산점도를 그립니다. 2개의 특성
  mglearn.discrete_scatter(X, y, ax=ax, s=2)
  ax.set_title("{}".format(title))
  ax.set_xlabel("특성 1")
  ax.set_ylabel("특성 2")

axes[0].legend(loc=3)
images.image.save_fig("5.ram_prices_plot_compare")  
plt.show()