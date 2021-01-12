# GroupBy 집계 메소드와 함수

import numpy as np
import pandas as pd

# df = pd.read_csv("syntax/test01.csv", index_col=0, encoding='UTF-8')
df = pd.read_csv("c:/data/test_combine_ab.txt")

print(df.shape)
print(df.columns)
print(df.info())
print(df)

# (1) GroupBy 메소드를 이용한 집계 (GroupBy aggregation using methods)\
# NA 값은 모두 무시되고 non-NA 값들에 대해서만 GroupBy method가 적용됩니다

# Making GroupBy object
# kvalues = df.groupby(['A001','A003','B002','B005','B006'])
# print(kvalues)

#print(kvalues.count())
#print(kvalues.count()['KEY'].describe())
#print(kvalues.count()['KEY'].max())
# 'A001','A003','B002','B005','B006' 에 대한 K-익명성 값
# print("K:",kvalues.count()['KEY'].min())

# subset
subset = df[['A001','A003','B002','B005','B006','KEY']]
#print(subset)

# 'A001','A003','B002','B005','B006' 에 대한 K-익명성 값
kvalue4 = subset.groupby(['A001','A003','B002','B005','B006'])

#print(kvalue4)
kValue = kvalue4.count()['KEY']
print(kValue, kValue.min())
print(kValue.shape)
#print(kValue.columns)
print(kValue.info())
kvalue.to_csv("C:/data/test_combine_ktest.txt")

kValueMin = kValue[np.where(kValue['KEY'] == kValue.min())]

print(kValueMin)
kValueMin.to_csv("C:/data/test_combine_ktest.txt")
print("K:",kvalue4.count()['KEY'].min())



