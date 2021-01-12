# 6. Database처럼 DataFrame Join/Merge 하기 : pd.merge()
import pandas as pd
from pandas import DataFrame as df
import numpy as np

'''
pd.merge(left, right, # merge할 DataFrame 객체 이름
         how='inner', # left, rigth, inner (default), outer
         on=None, # merge의 기준이 되는 Key 변수
         left_on=None, # 왼쪽 DataFrame의 변수를 Key로 사용
         right_on=None, # 오른쪽 DataFrame의 변수를 Key로 사용
         left_index=False, # 만약 True 라면, 왼쪽 DataFrame의 index를 merge Key로 사용
         right_index=False, # 만약 True 라면, 오른쪽 DataFrame의 index를 merge Key로 사용
         sort=True, # merge 된 후의 DataFrame을 join Key 기준으로 정렬
         suffixes=('_x', '_y'), # 중복되는 변수 이름에 대해 접두사 부여 (defaults to '_x', '_y'
         copy=True, # merge할 DataFrame을 복사
         indicator=False) # 병합된 이후의 DataFrame에 left_only, right_only, both 등의 출처를 알 수 있는 부가 정보 변수 추가
'''

df_left = pd.DataFrame({'KEY': ['K0', 'K1', 'K2', 'K3'],
   'A': ['A0', 'A1', 'A2', 'A3'],
   'B': ['B0', 'B1', 'B2', 'B3']})

df_right = pd.DataFrame({'KEY': ['K2', 'K3', 'K4', 'K5'],
   'C': ['C2', 'C3', 'C4', 'C5'],
   'D': ['D2', 'D3', 'D4', 'D5']})

print(df_left, df_right)

# (1) DataFrame을 key 기준으로 합치기 
# (1-1) Merge method : left (SQL join name : LEFT OUTER JOIN)
df_merge_how_left = pd.merge(df_left, df_right, how='left', on='KEY')
print(df_merge_how_left)

# (1-2) Merge method : right (SQL join name : RIGHT OUTER JOIN)
df_merge_how_right = pd.merge(df_left, df_right, how='right', on='KEY')
print(df_merge_how_right)

# (1-3) Merge method : inner (SQL join name : INNER JOIN)
df_merge_how_inner = pd.merge(df_left, df_right, 
   how='inner', # default
   on='KEY')
print(df_merge_how_right)

# (1-4) Merge method : outer (SQL join name : FULL OUTER JOIN)
df_merge_how_outer = pd.merge(df_left, df_right, how='outer', on='KEY')
print(df_merge_how_outer)

# (1-5) indicator = True : 병합된 이후의 DataFrame에 left_only, right_only, both 등의 출처를 알 수 있는 부가정보 변수 추가
#     _merge 새로운 변수
df_merge_indicator = pd.merge(df_left, df_right, how='outer', on='KEY', indicator=True)
print(df_merge_indicator)

# (1-6) 변수 이름이 중복될 경우 접미사 붙이기 : suffixes = ('_x', '_y')
#    'B'와 'C' 의 변수 이름이 동일하게 있는 두 개의 DataFrame을 만든 후에  KEY를 기준으로 합치기(merge)
#    변수 이름이 중복되므로 Data Source를 구분할 수 있도록 suffixes = ('string', 'string') 을 사용 
#    중복되는 변수의 뒷 부분에 접미사를 추가  default는 suffixes = ('_x', '_y') 
df_left_2 = pd.DataFrame({'KEY': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
    'C': ['C0', 'C1', 'C2', 'C3']})
df_right_2 = pd.DataFrame({'KEY': ['K0', 'K1', 'K2', 'K3'],
    'B': ['B0_2', 'B1_2', 'B2_2', 'B3_2'],
    'C': ['C0_2', 'C1_2', 'C2_2', 'C3_2'],
    'D': ['D0_2', 'D1_2', 'D2_2', 'D3_3']})

df_suffix = pd.merge(df_left_2, df_right_2, how='inner', on='KEY', suffixes=('_left', '_right'))
print(df_suffix)
# suffixes defaults to ('_x', '_y') 
df_suffix = pd.merge(df_left_2, df_right_2, how='inner', on='KEY')
print(df_suffix)

# (2) DataFrame을 index 기준으로 합치기 (merge, join on index)
#     pd.merge() 와 join() 두 가지 방법
df_left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
   'B': ['B0', 'B1', 'B2', 'B3']},
   index=['K0', 'K1', 'K2', 'K3'])
df_right = pd.DataFrame({'C': ['C2', 'C3', 'C4', 'C5'],
   'D': ['D2', 'D3', 'D4', 'D5']},
   index=['K2', 'K3', 'K4', 'K5'])

# (2-1) index를 기준으로 Left Join 하기 (Left join on index)
df_index1 = pd.merge(df_left, df_right, left_index=True, right_index=True, how='left')
print(df_index1)
df_index1 = df_left.join(df_right, how='left')
print(df_index1)

# (2-2) index를 기준으로 Right Join 하기 (Right join on index)
df_index2 = pd.merge(df_left, df_right, left_index=True, right_index=True, how='right')
print(df_index2)
df_left.join(df_right, how='right')
print(df_index2)

# (2-3) index를 기준으로 inner join 하기 (inner join on index)
df_index3 = pd.merge(df_left, df_right, left_index=True, right_index=True, how='inner')
print(df_index3)
df_index3 = df_left.join(df_right, how='inner')
print(df_index3)

# (2-4) index를 기준으로 outer join 하기 (outer join on index)
df_index4 = pd.merge(df_left, df_right, left_index=True, right_index=True, how='outer')
print(df_index4)
df_index4 = df_left.join(df_right, how='outer')
print(df_index4)

# (2-5) index와 Key를 혼합해서 DataFrame 합치기 (Joining key columns on an index)
df_left_2 = pd.DataFrame({'KEY': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']})
df_right_2 = pd.DataFrame({'C': ['C2', 'C3', 'C4', 'C5'],
    'D': ['D2', 'D3', 'D4', 'D5']},
    index=['K2', 'K3', 'K4', 'K5'])

df_index5 = pd.merge(df_left_2, df_right_2, left_on='KEY', right_index=True, how='left')
print(df_index5)
df_index5 = df_left_2.join(df_right_2, on='KEY', how='left')
print(df_index5)




