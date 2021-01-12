import numpy as np
import pandas as pd

people = {
    "first": ["Corey", 'Jane', 'John'], 
    "last": ["Schafer", 'Doe', 'Doe'], 
    "email": ["CoreyMSchafer@gmail.com", 'JaneDoe@email.com', 'JohnDoe@email.com']
}

print("people type: {}".format(type(people)))
print(people['email'])

df = pd.DataFrame(people)
df.info()
df.shape

print(df['email'])
print(df.email)
print("df type: {}".format(type(df['email'])))
print("df type: {}".format(type(df.email)))

print(df[['last', 'email']])

# Index(['first', 'last', 'email'], dtype='object')
print("df columns: {}".format(df.columns))

# 0    CoreyMSchafer@gmail.com
# 1          JaneDoe@email.com
print(df.iloc[[0,1],2])
# first                      Corey
# last                     Schafer
# email    CoreyMSchafer@gmail.com
print(df.loc[0])

#    first     last                    email
# 0  Corey  Schafer  CoreyMSchafer@gmail.com
# 1   Jane      Doe        JaneDoe@email.com
print(df.loc[[0,1]])

#                      email     last
# 0  CoreyMSchafer@gmail.com  Schafer
# 1        JaneDoe@email.com      Doe
print(df.loc[[0, 1], ['email', 'last']])

df.set_index('email')
print("df index: {}".format(df.index))
print("df inplace=false: {}".format(df))
print(df)
# inplace=True 
df.set_index('email', inplace=True)
print("df inplace=True: {}".format(df))
print("df index: {}".format(df.index))
df.reset_index(inplace=True)
print("df reset_index: {}".format(df))
print("df index: {}".format(df.index))

