# 3. DataFrame을 csv 파일로 내보내기 : df.to_csv()

import pandas as pd
from pandas import DataFrame

data = {'ID': ['A1', 'A2', 'A3', 'A4', 'A5'],
     'X1': [1, 2, 3, 4, 5],
     'X2': [3.0, 4.5, 3.2, 4.0, 3.5]}

data_df = DataFrame(data, index=['a', 'b', 'c', 'd', 'e']) # converting to DataFrame
print(data_df)

# 결측값(Missing Value)을 csv 파일로 내보낼 때 표기 지정하는 매개변수 설명을 위해서
# 제일 마지막 행(row)에 결측값을 추가
data_df_2 = data_df.reindex(['a', 'b', 'c', 'd', 'e', 'f'])
print(data_df)

data_df_2.to_csv('data/data_df_2.csv', # file path, file name
    sep=',',   # seperator, delimiter (구분자)
    na_rep='NaN')   # missing data representation (결측값 표기)

''' 디폴트 설정
header = True (첫번째 줄을 칼럼 이름으로 사용)
columns = 특정 칼럼만 csv 로 쓰기 (내보내기) 할 때 칼럼 이름을 list에 적어줌
index = True (행의 이름 index 도 같이 내보냄. index 내보내기 싫으면 False 명기)
float_format = '%.2f' (예: float8 을 소수점 둘째 자리까지 표기)
encoding = 'utf-8' (on Python 3)
line_terminator = '\n' (엔터로 줄 바꿈)
date_format = None (datetime 객체에 대한 format 설정하지 않음)
'''

data_df_2.to_csv('data/data_df_x2.csv',
    sep=',',
    na_rep='NaN', 
    float_format = '%.2f', # 2 decimal places
    columns = ['ID', 'X2'], # columns to write
    index = False) # do not write index



