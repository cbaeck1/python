import pandas as pd
import numpy as np


prem_league = pd.read_csv('basic/barchartrace/data/premierLeague_tables_1992-2017.csv')
prem_league.head()

prem_league = prem_league[['season', 'team', 'points']]
df = prem_league.pivot_table(values = 'points',index = ['season'], columns = 'team')

df.fillna(0, inplace=True)
df.sort_values(list(df.columns),inplace=True)
df = df.sort_index()
df.iloc[:, 0:-1] = df.iloc[:, 0:-1].cumsum()

top_prem_clubs = set()

for index, row in df.iterrows():
    top_prem_clubs |= set(row[row > 0].sort_values(ascending=False).head(6).index)

df = df[top_prem_clubs]

import bar_chart_race as bcr

bcr.bar_chart_race(df = df, 
                   n_bars = 6, 
                   sort='desc',
                   title='Premier League Clubs Points Since 1992',
                   filename = 'pl_clubs.mp4')