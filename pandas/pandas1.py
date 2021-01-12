# 1. csv 파일 불러오기
import pandas as pd

# 1. csv 파일 불러오기 : read_csv()
df = pd.read_csv('data/survey_results_public.csv')
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.columns)

schema_df = pd.read_csv('data/survey_results_schema.csv')
print(schema_df.shape)
print(schema_df.head(10))
print(schema_df.tail(10))
#print(schema_df.loc['MgrIdiot', 'QuestionText'])
#print(schema_df.sort_index(inplace=True))
#print(schema_df.head(10))
#print(schema_df.tail(10))

# 2. 구분자 '|' 인 text 파일 불러오기 : sep='|'
text_test = pd.read_csv('data/test_text_file.txt', sep='|')
print(text_test.shape)
print(text_test.columns)
print(text_test.head(10))

# 3. 파일 불러올 때 index 지정해주기 : index_col
df_index_col = pd.read_csv('data/survey_results_public.csv', index_col=0) # index_col='Respondent'
print(df_index_col.shape)
print(df_index_col.columns)
print(df_index_col.head(10))


#  4. 변수 이름(column name, header) 이 없는 파일 불러올 때 이름 부여하기
#     : names=['X1', 'X2', ... ], header=None
df_nohead = pd.read_csv('data/text_without_column_name.txt', sep='|', 
    names=['ID', 'A', 'B', 'C', 'D'], header=None, index_col='ID')
print(df_nohead.shape)
print(df_nohead.columns)
print(df_nohead.head(10))

# 5. 유니코드 디코드 에러, UnicodeDecodeError: 'utf-8' codec can't decode byte
f = pd.read_csv('data/test_text_file.txt', sep='|', encoding='CP949')

# 6. 특정 줄은 제외하고 불러오기: skiprows = [x, x]
csv_2 = pd.read_csv("data/survey_results_public.csv", skiprows = [1, 2])  
print(csv_2.shape)
print(csv_2.columns)
print(csv_2.head(10))

# 7. n 개의 행만 불러오기: nrows = n
csv_3 = pd.read_csv("data/survey_results_public.csv", nrows = 3)
print(csv_3.shape)
print(csv_3.columns)
print(csv_3.head(10))

# 8. 사용자 정의 결측값 기호 (custom missing value symbols) 
df_missing = pd.read_csv('data/test_text_file.txt', 
    na_values = ['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'])
print(df_missing.shape)
print(df_missing.columns)

# 9. 데이터 유형 설정 (Setting the data type per each column)
df_dtype = pd.read_csv('data/test_text_file.txt', 
    dtype = {"ID": int, 
             "LAST_NAME": str, 
             "AGE": float} )
print(df_dtype.shape)
print(df_dtype.columns)



'''
print(df['Hobbyist'])
print(df['email'])
print(df.loc[0:2, 'Hobbyist':'Employment'])
pd.set_option('display.max_columns', 85)
pd.set_option('display.max_rows', 85)
'''