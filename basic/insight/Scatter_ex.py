import matplotlib.pyplot as plt
import numpy as np

# 입력데이터 : numpy배열 같은 iterable 자료형
X = np.arange(20)
Y = np.random.randint(0, 20, 20)
S = np.abs(np.random.randn(20))*100
C = np.random.randint(0, 20, 20)
print(X,Y,S,c)

# plot 입력
# s 는 마커의 크기
# c 는 마커의 색상
scatter = plt.scatter(X, Y, s=S, c=C, label='A')

# X 및 Y 범위 설정
plt.xlim(X[0]-1, X[-1]+1)
plt.ylim(np.min(Y-1), np.max(Y+1))

# 그래픽 타이틀    x,y 축 라벨링
plt.title('Scatter Example', pad=10)
plt.xlabel('X axis', labelpad=10)
plt.ylabel('Y axis', labelpad=10)

# 틱설정
plt.xticks(np.linspace(X[0], X[-1], 11))
plt.yticks(np.linspace(np.min(np.append(Y, Y)), np.max(np.append(Y, Y)), 11))
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in', pad=8, top=True, right=True)

plt.show()