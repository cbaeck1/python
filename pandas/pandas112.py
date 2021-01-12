import pandas as pd


people = {
    "first": ["Corey", 'Jane', 'John'], 
    "last": ["Schafer", 'Doe', 'Doe'], 
    "email": ["CoreyMSchafer@gmail.com", 'JaneDoe@email.com', 'JohnDoe@email.com']
}
print(people['email']) # ['CoreyMSchafer@gmail.com', 'JaneDoe@email.com', 'JohnDoe@email.com']

df = pd.DataFrame(people)
print(df)

print("=========df['email']=========")
print(df['email'])
# 0    CoreyMSchafer@gmail.com
# 1          JaneDoe@email.com
# 2          JohnDoe@email.com
# Name: email, dtype: object
print(df.email)
print(df[['last', 'email']])
print(df.columns) # Index(['first', 'last', 'email'], dtype='object')
print(df.index) # RangeIndex(start=0, stop=3, step=1)

print('=========df.iloc[0]=========')
print(df.iloc[0]) 
# first                      Corey
# last                     Schafer
# email    CoreyMSchafer@gmail.com
# Name: 0, dtype: object
print('=========df.iloc[[0, 1], 2]=========')
print(df.iloc[[0, 1], 2])
# 0    CoreyMSchafer@gmail.com
# 1          JaneDoe@email.com
# Name: email, dtype: object
print(df.loc[[0, 1], ['email', 'last']])

print('=========set_index=========')
df.set_index('email')
print(df)
print(df.index)

df.set_index('email', inplace=True)
print(df)
print(df.index)

print()
print()




