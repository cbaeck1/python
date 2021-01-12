# GroupBy 집계 메소드와 함수

import numpy as np
import pandas as pd

# df = pd.read_csv("syntax/test01.csv", index_col=0, encoding='UTF-8')
df = pd.read_csv("c:/data/test_combine_ab.txt")

# (1) GroupBy 메소드를 이용한 집계 (GroupBy aggregation using methods)\
# NA 값은 모두 무시되고 non-NA 값들에 대해서만 GroupBy method가 적용됩니다

# subset
subset = df[['KEY','A001','A003','A007','A021','A066','A081','A082','A083','B002','B005','B006','B016','B039','B083','B100','B107','B108','B109']]
out = subset.set_index('KEY')
out.to_csv("C:/data/test_subset.txt")

