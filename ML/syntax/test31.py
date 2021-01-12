import pandas as pd

df = pd.read_csv('G:/work/workspace/ML/syntax/survey_results_public.csv', index_col='Respondent')
schema_df = pd.read_csv('G:/work/workspace/ML/syntax/survey_results_schema.csv', index_col='Column')

pd.set_option('display.max_columns', 85)
pd.set_option('display.max_rows', 85)

df.info()
print("df 5개 레코드\n: {}".format(df.head()))

print("df 크기: {}".format(df.shape))
print("df 속성: {}".format(df.columns))
print(df['Hobbyist'])
# row by Respondent, column
# 값이 없으면 KeyError: 0  
# print(df.loc[0:3, 'Hobbyist':'Employment'])
print(df.loc[1:3, 'Hobbyist':'Employment'])

schema_df.info()
print("schema_df 크기: {}".format(schema_df.shape))
print("schema_df 속성: {}".format(schema_df.columns))
print("df 5개 레코드\n: {}".format(schema_df.head()))

print(schema_df.loc['LanguageWorkedWith', 'QuestionText'])
schema_df.sort_index(inplace=True)
print("schema_df 5개 레코드\n: {}".format(schema_df.head()))
