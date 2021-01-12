# 5. 여러개의 동일한 형태 DataFrame 합치기 : pd.concat()
import pandas as pd
from pandas import DataFrame as df
import numpy as np

# (1) 여러개의 동일한 형태 DataFrame 합치기 : pd.concat()
# 데이터의 속성 형태가 동일한 데이터셋(homogeneously-typed objects)끼리 합칠 때
print('(1) 여러개의 동일한 형태 DataFrame 합치기 : pd.concat()')
'''
pd.concat(objs,  # Series, DataFrame, Panel object
             axis=0,  # 0: 위+아래로 합치기, 1: 왼쪽+오른쪽으로 합치기
             join='outer', # 'outer': 합집합(union), 'inner': 교집합(intersection)
             join_axes=None, # axis=1 일 경우 특정 DataFrame의 index를 그대로 이용하려면 입력
             ignore_index=False,  # False: 기존 index 유지, True: 기존 index 무시
             keys=None, # 계층적 index 사용하려면 keys 튜플 입력
             levels=None,
             names=None, # index의 이름 부여하려면 names 튜플 입력
             verify_integrity=False, # True: index 중복 확인
             copy=True) # 복사
'''

# (1-1) 위 + 아래로 DataFrame 합치기(rbind) : axis = 0
df_1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
   'B': ['B0', 'B1', 'B2'],
   'C': ['C0', 'C1', 'C2'],
   'D': ['D0', 'D1', 'D2']},
   index=[0, 1, 2])
df_2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
   'B': ['B3', 'B4', 'B5'],
   'C': ['C3', 'C4', 'C5'],
   'D': ['D3', 'D4', 'D5']},
   index=[3, 4, 5])

df_12_axis0 = pd.concat([df_1, df_2]) # row bind : axis = 0, default
print(df_12_axis0)

# (1-2) 왼쪽 + 오른쪽으로 DataFrame 합치기(cbind) : axis = 1
df_3 = pd.DataFrame({'E': ['A6', 'A7', 'A8'],
   'F': ['B6', 'B7', 'B8'],
   'G': ['C6', 'C7', 'C8'],
   'H': ['D6', 'D7', 'D8']},
   index=[0, 1, 2])

df_13_axis1 = pd.concat([df_1, df_3], axis=1) # column bind
print(df_13_axis1)

# (1-3) 합집합(union)으로 DataFrame 합치기 : join = 'outer'
df_4 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
   'B': ['B0', 'B1', 'B2'],
   'C': ['C0', 'C1', 'C2'],
   'E': ['E0', 'E1', 'E2']},
   index=[0, 1, 3])

df_14_outer = pd.concat([df_1, df_4], join='outer') # union, default
print(df_14_outer)

# (1-4) 교집합(intersection)으로 DataFrame 합치기 : join = 'inner'
df_14_inner = pd.concat([df_1, df_4], join='inner') # intersection
print(df_14_inner)

# (1-5) axis=1일 경우 특정 DataFrame의 index를 그대로 이용하고자 할 경우 : join_axes
# TypeError: concat() got an unexpected keyword argument 'join_axes'
# df_14_join_axes_axis1 = pd.concat([df_1, df_4], join_axes=[df_1.index], axis=1)
# print(df_14_join_axes_axis1)
df_14_outer_axis1 = pd.concat([df_1, df_4], join='outer', axis=1) # default
print(df_14_outer_axis1)

# (1-6) 기존 index를 무시하고 싶을 때 : ignore_index
df_5 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
   'B': ['B0', 'B1', 'B2'],
   'C': ['C0', 'C1', 'C2'],
   'D': ['D0', 'D1', 'D2']},
   index=['r0', 'r1', 'r2'])
df_6 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
   'B': ['B3', 'B4', 'B5'],
   'C': ['C3', 'C4', 'C5'],
   'D': ['D3', 'D4', 'D5']},
   index=['r3', 'r4', 'r5'])

df_56_with_index = pd.concat([df_5, df_6], ignore_index=False) # default
print(df_56_with_index)
df_56_ignore_index = pd.concat([df_5, df_6], ignore_index=True)# index 0~(n-1)
print(df_56_ignore_index)

# (1-7) 계층적 index (hierarchical index) 만들기 : keys 
df_56_with_keys = pd.concat([df_5, df_6], keys=['df_5', 'df_6'])
print(df_56_with_keys)
print(df_56_with_keys.loc['df_5'])
print(df_56_with_keys.loc['df_5'][0:2])

# (1-8) index에 이름 부여하기 : names
df_56_with_name = pd.concat([df_5, df_6],
   keys=['df_5', 'df_6'],
   names=['df_name', 'row_number'])
print(df_56_with_name)

# (1-9) index 중복 여부 점검 : verify_integrity
# df_7, df_8 DataFrame에 'r2' index를 중복으로 포함
df_7 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
   'B': ['B0', 'B1', 'B2'],
   'C': ['C0', 'C1', 'C2'],
   'D': ['D0', 'D1', 'D2']},
   index=['r0', 'r1', 'r2'])

df_8 = pd.DataFrame({'A': ['A2', 'A3', 'A4'],
   'B': ['B2', 'B3', 'B4'],
   'C': ['C2', 'C3', 'C4'],
   'D': ['D2', 'D3', 'D4']},
   index=['r2', 'r3', 'r4'])

df_78_F_verify_integrity = pd.concat([df_7, df_8], verify_integrity=False) # default
print(df_78_F_verify_integrity)

# Exception has occurred: ValueError Indexes have overlapping values: Index(['r2'], dtype='object')
try:
   df_78_T_verify_integrity = pd.concat([df_7, df_8], verify_integrity=True)
   print(df_78_T_verify_integrity)
except Exception as e:
   print("{}:{}".format(e.__class__, e))

# (2) DataFrame과 Series 합치기 : pd.concat(), append()
from pandas import Series
# (2-1) DataFrame에 Series '좌+우'로 합치기 : pd.concat([df, Series], axis=1)
df_1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
   'B': ['B0', 'B1', 'B2'],
   'C': ['C0', 'C1', 'C2'],
   'D': ['D0', 'D1', 'D2']},
   index=[0, 1, 2])
Series_1 = pd.Series(['S1', 'S2', 'S3'], name='S')

df_ds1 = pd.concat([df_1, Series_1], axis=1)
print(df_ds1)

# (2-2) DataFrame에 Series를 '좌+우'로 합칠 때
#      열 이름(column name) 무시하고 정수 번호 자동 부여 : ignore_index=True
df_ds2 = pd.concat([df_1, Series_1], axis=1, ignore_index=True)
print(df_ds2)

# (2-3) Series 끼리 '좌+우'로 합치기 : pd.concat([Series1, Series2, ...], axis=1)
# Series의 이름(name)이 있으면 합쳐진 DataFrame의 열 이름(column name)으로 사용
Series_1 = pd.Series(['S1', 'S2', 'S3'], name='S')
Series_2 = pd.Series([0, 1, 2]) # without name
Series_3 = pd.Series([3, 4, 5]) # without name
df_s1s2s3 = pd.concat([Series_1, Series_2, Series_3], axis=1)
print(df_s1s2s3)

# (2-4) Series 끼리 합칠 때 열 이름(column name) 덮어 쓰기 : keys = ['xx', 'xx', ...]
df_s1s2s3 = pd.concat([Series_1, Series_2, Series_3], axis=1, keys=['C0', 'C1', 'C1'])
print(df_s1s2s3)

# (2-5) DataFrame에 Series를 '위+아래'로 합치기 : df.append(Series, ignore_index=True)
Series_4 = pd.Series(['S1', 'S2', 'S3', 'S4'], index=['A', 'B', 'C', 'E'])
df_1.append(Series_4, ignore_index=True)
print(df_1)

# ignore_index=True 를 설정해주지 않으면 
try:
   df_1.append(Series_4) # TypeError without 'ignore_index=True'
   print(df_1)
except Exception as e:
   print("{}:{}".format(e.__class__, e))




