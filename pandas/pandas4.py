# 4. pd.DataFrame 만들고 Attributes 조회하기

import pandas as pd
from pandas import DataFrame as df
import numpy as np

'''
(1) pandas DataFrame 만들기
pd.DataFrame() 에서 사용하는 Parameter 들에는 (1) data, (2) index, (3) columns, (4) dtype, (5) copy 의 5가지가 있습니다.
(1-1) data : numpy ndarray, dict, DataFrame 등의 data source
(1-2) index : 행(row) 이름, 만약 명기하지 않으면 np.arange(n)이 자동으로 할당 됨
(1-3) column : 열(column) 이름, 만약 명기하지 않으면 역시 np.arnage(n)이 자동으로 할당 됨
(1-4) dtype : 데이터 형태(type), 만약 지정하지 않으면 Python이 자동으로 추정해서 넣어줌
(1-5) copy : 입력 데이터를 복사할지 지정. 디폴트는 False 임. (복사할 거 아니면 메모리 관리 차원에서 디폴트인 False 설정 사용하면 됨)
'''
print('(1) pandas DataFrame 만들기')
df_1 = df(data=np.arange(12).reshape(3, 4),
   index=['r0', 'r1', 'r2'], # Will default to np.arange(n) if no indexing
   columns=['c0', 'c1', 'c2', 'c3'],
   dtype='int', # Data type to force, otherwise infer
   copy=False) # Copy data from inputs
print(df_1)

'''
(2) DataFrame 의 Attributes 조회하기
(2-1) T : 행과 열 전치 (transpose)
(2-2) axes : 행과 열 이름을 리스트로 반환
(2-3) dtypes : 데이터 형태 반환
(2-4) shape : 행과 열의 개수(차원)을 튜플로 반환
(2-5) size : NDFrame의 원소의 개수를 반환
(2-6) values : NDFrame의 원소를 numpy 형태로 반환
'''
print('(2) DataFrame 의 Attributes 조회하기')
df_1.T # Transpose index and columns
df_1.axes
df_1.dtypes # Return the dtypes in this object
df_1.shape # Return a tuple representing the dimensionality of the DataFrame
df_1.size # number of elements in the NDFrame
df_1.values # Numpy representation of NDFrame

df_2 = pd.DataFrame({'class_1': ['a', 'a', 'b', 'b', 'c'],
    'var_1': np.arange(5),
    'var_2': np.random.randn(5)},
    index = ['r0', 'r1', 'r2', 'r3', 'r4'])
print(df_2)

# (3) 행(row) 기준으로 선택해서 가져오기
print('(3) 행(row) 기준으로 선택해서 가져오기')
# DataFrame의 index 를 확인
print(df_2.index)
# 'ix'를 사용하면 행 기준 indexing할 때 정수(int)와 행 이름(row label) 모두 사용할 수 있어서 편리
# Exception has occurred: AttributeError 'DataFrame' object has no attribute 'ix'
# Series.ix 및 DataFrame.ix 제거 (GH26438) 
# df.iloc[:, 'col_header']

print(df_2.iloc[2:]) # indexing from int. position to end
print(df_2.iloc[2]) # indexing specific row with int. position
print(df_2.loc['r2']) # indexing specific row with row label
print(df_2.head(2)) # Returns first n rows
print(df_2.tail(2)) # Returns last n rows

# (4) 열(column) 기준으로 선택해서 가져오기
print('(4) 열(column) 기준으로 선택해서 가져오기')
print(df_2.columns)
print(df_2['class_1'])
print(df_2[['class_1', 'var_1']])

# (5) index 재설정하기 (reindex)
print('(5) index 재설정하기')
idx = ['r0', 'r1', 'r2', 'r3', 'r4']
df_1 = pd.DataFrame({
    'c1': np.arange(5),
    'c2': np.random.randn(5)},
    index=idx)
print(df_1)
# (5-1) index 재설정하기 : reindex r3 -> r5, r4 -> r6
new_idx= ['r0', 'r1', 'r2', 'r5', 'r6']
df_1 = df_1.reindex(new_idx)
print(df_1)

# (5-2) reindex 과정에서 생긴 결측값 채우기 (fill in missing values) : fill_value
df_fill_value1 = df_1.reindex(new_idx, fill_value=0)
print(df_fill_value1)
df_fill_value2 = df_1.reindex(new_idx, fill_value='missing')
print(df_fill_value2)
df_fill_value3 = df_1.reindex(new_idx, fill_value='NA')
print(df_fill_value3)

# (6) 시계열 데이터 index 재설정
print('(6) 시계열 데이터 index 재설정')
# (6-1) 시계열 데이터 index 재설정 하기 (reindex of TimeSeries Data)
date_idx = pd.date_range('11/27/2016', periods=5, freq='D')
print(date_idx)
df_2 = pd.DataFrame({"c1": [10, 20, 30, 40, 50]}, index=date_idx)
print(df_2)
date_idx_2 = pd.date_range('11/25/2016', periods=10, freq='D')
df_timeseries = df_2.reindex(date_idx_2)
print(df_timeseries)
# (6-2) 시계열 데이터 reindex 과정에서 생긴 결측값 채우기 : method='ffill', 'bfill' (fill in missing value of TimeSeries Data)
df_ffill = df_2.reindex(date_idx_2, method='ffill') # forward-propagation
print(df_ffill)
df_bfill = df_2.reindex(date_idx_2, method='bfill') # back-propagation
print(df_bfill)

