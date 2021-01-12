# GroupBy 집계 메소드와 함수

import numpy as np
import pandas as pd

# df = pd.read_csv("syntax/test01.csv", index_col=0, encoding='UTF-8')
df = pd.read_csv("syntax/test01.csv")

print(df.shape)
print(df.columns)
print(df.info())
print(df)

# (1) GroupBy 메소드를 이용한 집계 (GroupBy aggregation using methods)\
# NA 값은 모두 무시되고 non-NA 값들에 대해서만 GroupBy method가 적용됩니다

# Making GroupBy object
# kvalues = df.groupby('SEX_TYPE') 
kvalues = df.groupby(['SEX_TYPE','AGE'])
print(kvalues)

print(kvalues.count())
print(kvalues.count()['ASK_ID'].describe())
print(kvalues.count()['ASK_ID'].max())
# SEX_TYPE, AGE 에 대한 K-익명성 값
print(kvalues.count()['ASK_ID'].min())

# STD_YYYY, SEX_TYPE, GAIBJA_TYPE 에 대한 K-익명성 값
kvalue2 = df.groupby(['STD_YYYY','SEX_TYPE','GAIBJA_TYPE'])
print(kvalue2)
print(kvalue2.count()['ASK_ID'].min())

# STD_YYYY, SEX_TYPE, GAIBJA_TYPE,RVSN_ADDR_CD 에 대한 K-익명성 값
kvalue3 = df.groupby(['STD_YYYY','SEX_TYPE','GAIBJA_TYPE','RVSN_ADDR_CD'])
print(kvalue3)
print(kvalue3.count()['ASK_ID'].min())


# subset
subset = df[['STD_YYYY','SEX_TYPE','GAIBJA_TYPE','RVSN_ADDR_CD','ASK_ID']]
print(subset)

# STD_YYYY, SEX_TYPE, GAIBJA_TYPE,RVSN_ADDR_CD 에 대한 K-익명성 값
kvalue4 = subset.groupby(['STD_YYYY','SEX_TYPE','GAIBJA_TYPE','RVSN_ADDR_CD'])
print(kvalue4)
print(kvalue4.count()['ASK_ID'])
print(kvalue4.count()['ASK_ID'].min())

#  SEX_TYPE, GAIBJA_TYPE,RVSN_ADDR_CD 에 대한 K-익명성 값
kvalue5 = subset.groupby(['SEX_TYPE','GAIBJA_TYPE','RVSN_ADDR_CD'])
kvalue5.count()['ASK_ID'].to_csv("syntax/kvalue5.csv")
print(kvalue5)
print(kvalue5.count()['ASK_ID'])
print(kvalue5.count()['ASK_ID'].min())

