import pandas as pd
import glob, os, csv

# pd.merge(df1, df2, on = 'asdf' how = 'inner' or 'outer' left right)
csv1 = pd.read_csv("C:/data/test_data_a.txt")
print("csv1", csv1.info())
csv2 = pd.read_csv("C:/data/test_data_b.txt")
print("csv2", csv2.info())

# csv3 = pd.merge(dataframe_A, dataframe_B[0:], left_on = 'KEY',right_on = 'KEY' ,how = 'inner')
csv3 = pd.merge(csv1, csv2[0:], left_on = 'KEY',right_on = 'KEY' ,how = 'inner')
out = csv3.set_index('KEY')
print("csv3", csv3.info())
print("out", out.info())
out.to_csv("C:/data/test_combine_ab.txt")
