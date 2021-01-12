import numpy as np # numpy 도 함께 import
import pandas as pd

# 1. Pandas 자료구조
# Pandas에서는 기본적으로 정의되는 자료구조인 Series와 Data Frame을 사용합니다.
# 이 자료구조들은 빅 데이터 분석에 있어서 높은 수준의 성능을 보여줍니다.
# 1-1. Series
# Series 정의하기
obj = pd.Series([4, 7, -5, 3])
print("obj:", obj)
# Series의 값만 확인하기
print("obj.values:", obj.values)
# Series의 인덱스만 확인하기
print("obj.index:", obj.index)
# Series의 자료형 확인하기
print("obj.dtypes:", obj.dtypes)
# 인덱스를 바꿀 수 있다.
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print("obj2:", obj2)
# python의 dictionary 자료형을 Series data로 만들 수 있다.
# dictionary의 key가 Series의 index가 된다
sdata = {'Kim': 35000, 'Beomwoo': 67000, 'Joan': 12000, 'Choi': 4000}
obj3 = pd.Series(sdata)
print("obj3:", obj3)
# index 변경
obj3.index = ['A', 'B', 'C', 'D']
print("obj3:", obj3)

# 1.2. Data Frame
# Data Frame 정의하기
# 먼저 DataFrame에 들어갈 데이터를 정의해주어야 하는데, python의 dictionary 또는 numpy 의 array 로 정의할 수 있다.
data = {'name': ['Beomwoo', 'Beomwoo', 'Beomwoo', 'Kim', 'Park'],
        'year': [2013, 2014, 2015, 2016, 2015],
        'points': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)
print("df:\n", df)

# 행과 열의 구조를 가진 데이터가 생긴다.
# 행 방향의 index
print("df.index:", df.index) # RangeIndex(start=0, stop=5, step=1)
# 열 방향의 index
print("df.columns:", df.columns) # Index(['name', 'year', 'points'], dtype='object')
# 값 얻기
print("df.values:", df.values)
# array([['Beomwoo', 2013, 1.5],
#        ['Beomwoo', 2014, 1.7],
#        ['Beomwoo', 2015, 3.6],
#        ['Kim', 2016, 2.4],
#        ['Park', 2015, 2.9]], dtype=object)
# 각 인덱스에 대한 이름 설정하기
df.index.name = 'Num'
df.columns.name = 'Info'
print("df:\n", df)

# DataFrame을 만들면서 columns와 index를 설정할 수 있다.
df2 = pd.DataFrame(data, columns=['year', 'name', 'points', 'penalty'],
                         index=['one', 'two', 'three', 'four', 'five'])
print("df2:\n", df2)

# DataFrame 을 정의하면서, data 로 들어가는 python dictionary 와 columns 의 순서가 달라도 알아서 맞춰서 정의된다.
# 하지만 data 에 포함되어 있지 않은 값은 NaN(Not a Number)으로 나타나게 되는데, 이는 null과 같은 개념이다.
# NaN값은 추후에 어떠한 방법으로도 처리가 되지 않는 데이터이다.
# 따라서 올바른 데이터 처리를 위해 추가적으로 값을 넣어줘야 한다.
# describe() 함수는 DataFrame의 계산 가능한 값들에 대한 다양한 계산 값을 보여준다.
print("df.describe():", df.describe())

# 2. DataFrame Indexing
data = {"names": ["Kilho", "Kilho", "Kilho", "Charles", "Charles"],
        "year": [2014, 2015, 2016, 2015, 2016],
        "points": [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data, columns=["year", "names", "points", "penalty"],
                        index=["one", "two", "three", "four", "five"])
print("df:\n", df)
# 2-1. DataFrame에서 열을 선택하고 조작하기
print("df['year']:", df['year'])
print("df.year:", df.year)

print("df[['year','points']]:", df[['year','points']])
# 특정 열에 대해 위와 같이 선택하고, 값을 변경
df['penalty'] = 0.5
print("df:\n", df)
# 또는
df['penalty'] = [0.1, 0.2, 0.3, 0.4, 0.5] # python의 List나 numpy의 array
# 새로운 열을 추가하기
df['zeros'] = np.arange(5)
# Series를 추가할 수도 있다.
val = pd.Series([-1.2, -1.5, -1.7], index=['two','four','five'])
df['debt'] = val
# 하지만 Series로 넣을 때는 val와 같이 넣으려는 data의 index에 맞춰서 데이터가 들어간다.
# 이점이 python list나 numpy array로 데이터를 넣을때와 가장 큰 차이점이다.
df['net_points'] = df['points'] - df['penalty']
df['high_points'] = df['net_points'] > 2.0
print("df 열 삭제 전:\n", df)

# 열 삭제하기 
del df['high_points']
del df['net_points']
del df['zeros']
print("df 열 삭제 후:\n", df)

print("df.columns:", df.columns) # Index(['year', 'names', 'points', 'penalty', 'debt'], dtype='object')
df.index.name = 'Order'
df.columns.name = 'Info'
print("df:\n", df)

# 컬럼 삭제 : drop 명령어를 통해 컬럼 전체를 삭제할 수 있다. axis=1은 컬럼을 뜻한다. 
# axis=0인 경우, 로우를 삭제하며 이것이 디폴트이다. 
# inplace의 경우 drop한 후의 데이터프레임으로 기존 데이터프레임을 대체하겠다는 뜻이다. 
# 즉, 아래의 inplace=True는 df = df.drop('A', axis=1)과 같다.
df_column = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['A', 'B', 'C'])
print("df_column:\n", df_column)
# Drop the column with label 'A', drop axis의 경우 column이면 1, row이면 0이다.
df_column.drop('A', axis=1, inplace=True)
print("df_column:\n", df_column)

# 중복 로우 삭제 : 중복 로우 중에 어떤것을 남길것인가 정할 수 있다.
df_dup = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [40, 50, 60], [23, 35, 37]]), 
                  index= [2.5, 12.6, 4.8, 4.8, 2.5], 
                  columns=[48, 49, 50])
print("df_dup:\n", df_dup)
df_dup = df_dup.reset_index()  # 0 ~ index 추가
print("df_dup:\n", df_dup)
df_dup = df_dup.drop_duplicates(subset='index', keep='last').set_index('index') # 3    4.8  40  50  60 을 남김
print("df_dup:\n", df_dup)

# 2.2. DataFrame에서 행을 선택하고 조작하기
# index, slice
# 0번째 부터 2(3-1) 번째까지 가져온다. 뒤에 써준 숫자번째의 행은 뺀다.
print("df[0:3]:", df[0:3])
# tow라는 행부터 four라는 행까지 가져온다. 뒤에 써준 이름의 행을 빼지 않는다.
print("df['two':'four'] :",df['two':'four']) # 하지만 비추천!
# .loc 또는 .iloc 함수를 사용하는 방법을 권장 : 반환은 Series
print("df.loc['two']:\n", df.loc['two'])
print("df.loc['two':'four']:\n", df.loc['two':'four'])
print("df.loc['two':'four', 'points']:\n", df.loc['two':'four', 'points'])
print("df.loc[:,'year']:", df.loc[:,'year'])  # == df['year']

# 새로운 행 삽입하기
df.loc['six',:] = [2013,'Jun',4.0,0.1,2.1]
print("df:\n", df)

# .iloc 사용:: index 번호를 사용한다.
print("df.iloc[3]:", df.iloc[3]) # 3번째 행을 가져온다.
print("df.iloc[3:5, 0:2]:", df.iloc[3:5, 0:2])
print("df.iloc[[0,1,3], [1,2]]:", df.iloc[[0,1,3], [1,2]])
print("df.iloc[:,1:4]:", df.iloc[:,1:4])
print("df.iloc[1,1]:", df.iloc[1,1])

# 3. DataFrame에서의 boolean Indexing
# year가 2014보다 큰 boolean data
print("df['year'] > 2014:", df['year'] > 2014)
# year가 2014보다 큰 모든 행의 값
print("df.loc[df['year']>2014,:]:", df.loc[df['year']>2014,:])
print("df.loc[df['names'] == 'Kilho',['names','points']]:", df.loc[df['names'] == 'Kilho',['names','points']] )
# numpy에서와 같이 논리연산을 응용할 수 있다.
print("df.loc[(df['points']>2)&(df['points']<3),:]:", df.loc[(df['points']>2)&(df['points']<3),:])
# 새로운 값을 대입할 수도 있다.
df.loc[df['points'] > 3, 'penalty'] = 0

# 4. Data
# DataFrame을 만들때 index, column을 설정하지 않으면 기본값으로 0부터 시작하는 정수형 숫자로 입력된다.
df = pd.DataFrame(np.random.randn(6,4))
print("df:\n", df)

df.columns = ['A', 'B', 'C', 'D']
print("df.columns:", df.columns)
df.index = pd.date_range('20160701', periods=6)
# pandas에서 제공하는 date range함수는 datetime 자료형으로 구성된, 날짜 시각등을 알 수 있는 자료형을 만드는 함수
print("df.index:", df.index)
# np.nan은 NaN값을 의미한다.
df['F'] = [1.0, np.nan, 3.5, 6.1, np.nan, 7.0]
print("df:\n", df)
# NaN 없애기 : 행의 값중 하나라도 nan인 경우 그 행을 없앤다.
df.dropna(how='any') 
print("df:\n", df)

# 행의 값의 모든 값이 nan인 경우 그 행으 없앤다.
df.dropna(how='all')
print("df:\n", df)

# 주의 drop함수는 특정 행 또는 열을 drop하고난 DataFrame을 반환한다.
# 즉, 반환을 받지 않으면 기존의 DataFrame은 그대로이다.
# 아니면, inplace=True라는 인자를 추가하여, 반환을 받지 않고서도 기존의 DataFrame이 변경되도록 한다.
# nan값에 값 넣기
df.fillna(value=0.5)
# nan값인지 확인하기
print("df.isnull():", df.isnull())
# F열에서 nan값을 포함하는 행만 추출하기
print("df.loc[df.isnull()['F'],:]:", df.loc[df.isnull()['F'],:])

pd.to_datetime('20160701')
# 특정 행 drop하기
df.drop(pd.to_datetime('20160701'))
# 2개 이상도 가능
df.drop([pd.to_datetime('20160702'), pd.to_datetime('20160704')])
print("df:\n", df)
# 특정 열 삭제하기
df.drop('F', axis = 1)
# 2개 이상의 열도 가능
df.drop(['B','D'], axis = 1)
print("df:\n", df)


# 5. Data 분석용 함수들
data = [[1.4, np.nan],
        [7.1, -4.5],
        [np.nan, np.nan],
        [0.75, -1.3]]
df_analysis = pd.DataFrame(data, columns=["one", "two"], index=["a", "b", "c", "d"])
print("df_analysis:\n", df_analysis)
# 행방향으로의 합(즉, 각 열의 합)
print("df_analysis.sum(axis=0):", df_analysis.sum(axis=0))
# 열방향으로의 합(즉, 각 행의 합)
print("df_analysis.sum(axis=1):", df_analysis.sum(axis=1))
# 이때, 위에서 볼 수 있듯이 NaN값은 배제하고 계산한다.
# NaN 값을 배제하지 않고 계산하려면 아래와 같이 skipna에 대해 false를 지정해준다.
print("df_analysis.sum(axis=1, skipna=False):", df_analysis.sum(axis=1, skipna=False))

# 특정 행 또는 특정 열에서만 계산하기
print("df_analysis['one'].sum():", df_analysis['one'].sum())
print("df_analysis.loc['b'].sum():", df_analysis.loc['b'].sum())

'''
pandas에서 DataFrame에 적용되는 함수들
sum() 함수 이외에도 pandas에서 DataFrame에 적용되는 함수는 다음의 것들이 있다.
count 전체 성분의 (NaN이 아닌) 값의 갯수를 계산
min, max 전체 성분의 최솟, 최댓값을 계산
argmin, argmax 전체 성분의 최솟값, 최댓값이 위치한 (정수)인덱스를 반환
idxmin, idxmax 전체 인덱스 중 최솟값, 최댓값을 반환
quantile 전체 성분의 특정 사분위수에 해당하는 값을 반환 (0~1 사이)
sum 전체 성분의 합을 계산
mean 전체 성분의 평균을 계산
median 전체 성분의 중간값을 반환
mad 전체 성분의 평균값으로부터의 절대 편차(absolute deviation)의 평균을 계산
std, var 전체 성분의 표준편차, 분산을 계산
cumsum 맨 첫 번째 성분부터 각 성분까지의 누적합을 계산 (0에서부터 계속 더해짐)
cumprod 맨 첫번째 성분부터 각 성분까지의 누적곱을 계산 (1에서부터 계속 곱해짐)
'''

df_analysis2 = pd.DataFrame(np.random.randn(6, 4),
                   columns=["A", "B", "C", "D"],
                   index=pd.date_range("20160701", periods=6))
print("df_analysis2:\n", df_analysis2)

# A열과 B열의 상관계수 구하기
print("df_analysis2['A'].corr(df2['B']):", df_analysis2['A'].corr(df_analysis2['B']))  # -0.06715327766901227
# B열과 C열의 공분산 구하기
print("df_analysis2['B'].cov(df2['C']):", df_analysis2['B'].cov(df_analysis2['C'])) # -1.0099019967454226
# 정렬함수 및 기타함수
dates = df_analysis2.index
random_dates = np.random.permutation(dates)
df_analysis2 = df_analysis2.reindex(index=random_dates, columns=["D", "B", "C", "A"])
print("df_analysis2:\n", df_analysis2)
# index와 column의 순서가 섞여있다.
# 이때 index가 오름차순이 되도록 정렬해보자
df_analysis2.sort_index(axis=0)
# column을 기준으로?
df_analysis2.sort_index(axis=1)
# 내림차순으로는?
df_analysis2.sort_index(axis=1, ascending=False)
# 값 기준 정렬하기 : D열의 값이 오름차순이 되도록 정렬하기
df_analysis2.sort_values(by='D')
# B열의 값이 내림차순이 되도록 정렬하기
df_analysis2.sort_values(by='B', ascending=False)
df_analysis2["E"] = np.random.randint(0, 6, size=6)
df_analysis2["F"] = ["alpha", "beta", "gamma", "gamma", "alpha", "gamma"]
print("df_analysis2:\n", df_analysis2)
# E열과 F열을 동시에 고려하여, 오름차순으로 하려면?
df_analysis2.sort_values(by=['E','F'])
# 지정한 행 또는 열에서 중복값을 제외한 유니크한 값만 얻기
print("df_analysis2['F'].unique():\n", df_analysis2['F'].unique()) # array(['alpha', 'beta', 'gamma'], dtype=object)
# 지정한 행 또는 열에서 값에 따른 개수 얻기
print("df_analysis2['F'].value_counts():\n", df_analysis2['F'].value_counts())
# 지정한 행 또는 열에서 입력한 값이 있는지 확인하기
# F열의 값이 alpha나 beta인 모든 행 구하기
print("df_analysis2['F'].isin(['alpha','beta']):\n", df_analysis2['F'].isin(['alpha','beta']))

# 사용자가 직접 만든 함수를 적용하기
df_user = pd.DataFrame(np.random.randn(4, 3), 
                   columns=["b", "d", "e"],
                   index=["Seoul", "Incheon", "Busan", "Daegu"])
print("df_user:\n", df_user)

func = lambda x: x.max() - x.min()
print("df_user.apply(func, axis=0):\n", df_user.apply(func, axis=0))

# 열 또는 행에 함수 적용하기
df4 = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['A', 'B', 'C'])
doubler = lambda x: x*2
print(df4.head())
print(df4['A'].map(doubler))
print(df4.apply(doubler))
print(df4.iloc[1].apply(doubler))

# 빈 데이터 프레임 만들기 (Creating empty dataframe)
# 가끔 빈 데이터 프레임을 만들고, 그 곳에 값을 채워야할 경우가 있다. 이것을 하기 위해서는 인덱스와 컬럼을 만든 후, np.nan 을 채워주면 된다.
df_empty = pd.DataFrame(np.nan, index=[0,1,2,3], columns=['A'])
print("df_empty:\n", df_empty)
# 그 속에 값을 채워넣고 싶으면 아래처럼 할 수 있다.
df_empty.loc[1, 'A'] = "A"

# 데이터 임포트시, 날짜, 시간 parsing 하기
# 만약 날짜 시간으로된 컬럼을 datatime 으로 parsing 하기 위해서는 read_csv 메소드의 parse_dates 를 이용할 수 있다.
from datetime import datetime
dateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
# Which makes your read command:
df_date1 = pd.read_csv('data/date_sample', parse_dates=['datetime'], date_parser=dateparser)
print("df_date1:\n", df_date1)
# 여러 열을 단일 날짜 시간 열로 결합 할 수도 있습니다. 'date'와 'time'열을 단일 'datetime'열로 병합
df_date2 = pd.read_csv('data/date_sample2', parse_dates={'datetime': ['date', 'time']}, date_parser=dateparser)  
print("df_date2:\n", df_date2)

# 데이터프레임 재구성하기
# pivot 함수를 통해 index, column, values 를 지정하여 재구성할 수 있다.
products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 'Entertainment', 'Tech', 'Tech'],
        'store': ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
        'price':[11.42, 23.50, 19.99, 15.95, 55.75, 111.55],
        'testscore': [4, 3, 5, 7, 5, 8]})
print("products:\n", products)        
pivot_products = products.pivot(index='category', columns='store', values='price')
print("pivot_products:\n", pivot_products)        

# stack & Unstack
# pivot 함수를 이용할 수도 있지만, stack 과 unstack 함수를 이용하여 pivoting 을 할 수 있다. 
# stack 은 column index 를 raw index 로 바꾸어 데이터프레임이 길어지고, 
# unstack 은 raw index 를 column index 로 바꾸어 데이터프레임이 넓어진다. 
# Stack 과 Unstack 의 개념도
# https://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures 참고

# stack : df.stack()
#     c0 c1 c2           c0 1
# r0  1  2  3    ==>  r0 c1 2
# r1  4  5  6            c2 3
#                     r1 c0 4
#                        c1 5
#                        c2 6
# unstack : df.unstack()
#     c0 c1 c2        c0    c1    c2 
# r0  1  2  3    ==>  r0 r1 r0 r1 r0 r1
# r1  4  5  6         1  2  3  4  5  6
stack_products = products.stack()
print("stack_products:\n", stack_products)      
restore_products = stack_products.unstack()
print("restore_products:\n", restore_products)     
unstack_products = products.unstack()
print("unstack_products:\n", unstack_products)     

# index 를 바꾸면 unstack 으로 복원되지 않음
'''
products.set_index('category', inplace=True)
print("products:\n", products)      
stack_products2 = products.stack()
print("stack_products2:\n", stack_products2)      
unstack_products2 = products.unstack()
print("unstack_products2:\n", unstack_products2)     
restore_products2 = stack_products2.unstack()
print("restore_products2:\n", restore_products2)     
'''

# Row Multi-Index
row_idx_arr = list(zip(['r0', 'r0'], ['r-00', 'r-01']))
row_idx = pd.MultiIndex.from_tuples(row_idx_arr)
# Column Multi-Index
col_idx_arr = list(zip(['c0', 'c0', 'c1'], ['c-00', 'c-01', 'c-10']))
col_idx = pd.MultiIndex.from_tuples(col_idx_arr)
# Create the DataFrame
df_org = pd.DataFrame(np.arange(6).reshape(2,3), index=row_idx, columns=col_idx)
df_org = df_org.applymap(lambda x: (x // 3, x % 3))
print("df_org:\n", df_org)   
# Stack/Unstack
df_stack = df_org.stack()
print("df_stack:\n", df_stack)   
df_unstack = df_org.unstack()
print("df_unstack:\n", df_unstack)   

# melt 함수 : column 을 row로 바꾸어준다. 
# 녹으면 흘러내리니까 column 이 raw로 흘러려 column 이 짧아지고 raw 는 길어진다고 이해하면 쉽다. 
# The `people` DataFrame
people = pd.DataFrame({'FirstName' : ['John', 'Jane'],
                       'LastName' : ['Doe', 'Austen'],
                       'BloodType' : ['A-', 'B+'],
                       'Weight' : [90, 64]})
print("people:\n", people)   
# Use `melt()` on the `people` DataFrame
people_melt = pd.melt(people, id_vars=['FirstName', 'LastName'], var_name='measurements')
print("people_melt:\n", people_melt)   

# 데이터 프레임 반복하기
# iterrows 함수를 이용하여 iteration 할 수 있다. 매우 유용하다.
df_iter = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['A', 'B', 'C'])
for index, row in df_iter.iterrows() :
    print(row['A'], row['B'])
print("df_iter:\n", df_iter)        


'''

print(" :", )
print(" :", )
print(" :", )
print(" :", )
print(" :", )
print(" :", )
print(" :", )

print(" :"\n, )
print(" :"\n, )
print(" :"\n, )
print(" :"\n, )
print(" :"\n, )
print(" :"\n, )
print(" :"\n, )
print(" :"\n, )
print(" :", )
print("", )
print("", )
print()
print()
'''



