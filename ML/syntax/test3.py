import pandas as pd

df = pd.read_csv('G:/work/workspace/ML/syntax/survey_results_public.csv')
schema_df = pd.read_csv('G:/work/workspace/ML/syntax/survey_results_schema.csv')

# pd.set_option('display.max_columns', 85)
# pd.set_option('display.max_rows', 85)

df.info()
print("df 5개 레코드\n: {}".format(df.head()))

print("df 크기: {}".format(df.shape))
print("df 속성: {}".format(df.columns))
print(df['Hobbyist'])
# row by index, column
print(df.loc[0:2, 'Hobbyist':'Employment'])

schema_df.info()
print("schema_df 크기: {}".format(schema_df.shape))
print("schema_df 속성: {}".format(schema_df.columns))
print("df 5개 레코드\n: {}".format(schema_df.head()))

