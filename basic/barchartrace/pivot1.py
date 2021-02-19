import matplotlib.pyplot as plt
import bar_chart_race as bcr

# df = bcr.load_dataset('covid19_tutorial')
df = bcr.load_dataset('COVID-19')
print(df.head())

# 
current_day = '2020-12-14'
# dff = (df[df['dateRep'].eq(current_day)]
#        .sort_values(by='cases', ascending=False)
#        .head(10))
# print(dff)

# fig, ax = plt.subplots(figsize=(15, 8))
# ax.barh(dff['countriesAndTerritories'], dff['cases'])
# plt.show()

# data_pivot = df.pivot(index='dateRep', columns='countriesAndTerritories', values='cases')
# print(data_pivot)

# 누적합
# print (df.groupby(by=['dateRep','countriesAndTerritories']).sum().groupby(level=[0]).cumsum())
# print (df.groupby(by=['dateRep','countriesAndTerritories']).sum()['cases'].cumsum())

# 최종누적합
df_sum0 = df.groupby(by=['countriesAndTerritories']).cumsum()
print(df_sum0)

# 누적합 :일자, 나라
df_sum = df.groupby(by=['date','countriesAndTerritories']).sum().groupby(level=[1]).cumsum()
print(df_sum)

df_sum = df_sum.reset_index(drop=False)
print(df_sum.head())

dff = (df_sum[df_sum['date'].eq(current_day)]
       .sort_values(by='cases', ascending=False)
       .head(10))
print(dff)

fig, ax = plt.subplots(figsize=(15, 8))
ax.barh(dff['countriesAndTerritories'], dff['cases'])
plt.show()

data_pivot0 = dff.pivot(index='date', columns='countriesAndTerritories', values='cases')
print(data_pivot0)

data_pivot = df_sum.pivot(index='date', columns='countriesAndTerritories', values='cases')
#data_pivot = data_pivot.set_index('date', drop=True)
print(data_pivot.info(), len(data_pivot.index),len(data_pivot.columns))
print(data_pivot)
