import pandas as pd

##########데이터 로드

df = pd.DataFrame([
        ['A01', 2, 1, 60, 139, 'country', 0, '10,000', 3, 3],
        ['A02', 3, 2, 80, 148, 'country', 0, '12,000', 5, 4],
        ['A03', 3, 4, 50, 149, 'country', 0, '13,000', 7, 8],
        ['A04', 5, 5, 40, 151, 'country', 0, '11,000', 10, 11],
        ['A05', 7, 5, 35, 154, 'city', 0, '20,000', 12, 11],
        ['A06', 2, 5, 45, 149, 'country', 0, '30,000', 7, 6],
        ['A07',8, 9, 40, 155, 'city', 1, '10,000', 13, 12],
        ['A08', 9, 10, 70, 155, 'city', 3, '13,000', 13, 14],
        ['A09', 6, 12, 55, 154, 'city', 0, '30,000', 12, 13],
        ['A10', 9, 2, 40, 156, 'city', 1, '15,000', 13, 12],
        ['A11', 6, 10, 60, 153, 'city', 0, '12,000', 12, 13],
        ['A12', 2, 4, 75, 151, 'country', 0, '13,000', 6, 7]
    ], columns=['ID', 'hour', 'attendance', 'weight', 'iq', 'region', 'library', 'money', 'english', 'math'])

print(df)

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

# 한글폰트 사용 방법2
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location = "c:/Windows/Fonts/malgun.ttf" # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font', family=font_name)

# 바 차트
y = [2, 3, 1]
x = np.arange(len(y))
xlabel = ['가', '나', '다']
plt.title("Bar Chart")
plt.bar(x, y)
plt.xticks(x, xlabel)
plt.yticks(sorted(y))
plt.xlabel("가나다")
plt.ylabel("빈도 수")
plt.show()

# 스템 플롯
# 주로 이산 확률 함수나 자기상관관계(auto-correlation)를 묘사할 때 사용
x = np.linspace(0.1, 2 * np.pi, 10)
plt.title("Stem Plot")
plt.stem(x, np.cos(x), '-.')
plt.show()

# 파이 차트
labels = ['개구리', '돼지', '개', '통나무']
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)
plt.title("Pie Chart")
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

# 히스토그램
# hist 명령은 bins 인수로 데이터를 집계할 구간 정보를 받는다
np.random.seed(0)
x = np.random.randn(1000)
plt.title("Histogram")
arrays, bins, patches = plt.hist(x, bins=10)
plt.show()

# 스캐터 플롯
np.random.seed(0)
X = np.random.normal(0, 1, 100)
Y = np.random.normal(0, 1, 100)
plt.title("Scatter Plot")
plt.scatter(X, Y)
plt.show()


N = 30
np.random.seed(0)
x = np.random.rand(N)
y1 = np.random.rand(N)
y2 = np.random.rand(N)
y3 = np.pi * (15 * np.random.rand(N))**2
plt.title("Bubble Chart")
plt.scatter(x, y1, c=y2, s=y3)
plt.show()

# Imshow

from sklearn.datasets import load_digits
digits = load_digits()
X = digits.images[0]
print(X)

plt.title("mnist digits; 0")
plt.imshow(X, interpolation='nearest', cmap=plt.cm.bone_r)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.subplots_adjust(left=0.35, right=0.65, bottom=0.35, top=0.65)
plt.show()

# 데이터 수치를 색으로 바꾸는 함수는 칼라맵(color map)이라고 한다.
# 칼라맵은 cmap 인수로 지정한다. 사용할 수 있는 칼라맵은 plt.cm의 속성으로 포함되어 있다

fig, axes = plt.subplots(1, 4, figsize=(12, 3),
                         subplot_kw={'xticks': [], 'yticks': []})
axes[0].set_title("plt.cm.Blues")
axes[0].imshow(X, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title("plt.cm.Blues_r")
axes[1].imshow(X, interpolation='nearest', cmap=plt.cm.Blues_r)
axes[2].set_title("plt.BrBG")
axes[2].imshow(X, interpolation='nearest', cmap='BrBG')
axes[3].set_title("plt.BrBG_r")
axes[3].imshow(X, interpolation='nearest', cmap='BrBG_r')
plt.show()

methods = [
    None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
]
fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(X, cmap=plt.cm.bone_r, interpolation=interp_method)
    ax.set_title(interp_method)
plt.show()

# 컨투어 플롯
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
XX, YY = np.meshgrid(x, y)
ZZ = f(XX, YY)

plt.title("Contour plots")
plt.contourf(XX, YY, ZZ, alpha=.75, cmap='jet')
plt.contour(XX, YY, ZZ, colors='black')
plt.show()


# 3D 서피스 플롯

from mpl_toolkits.mplot3d import Axes3D
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
XX, YY = np.meshgrid(X, Y)
RR = np.sqrt(XX**2 + YY**2)
ZZ = np.sin(RR)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title("3D Surface Plot")
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='hot')
plt.show()




