import pandas_alive
import pandas as pd

# Data Source: https://ourworldindata.org/grapher/population-by-country
df = pd.read_csv('basic/barchartrace/data/population-by-country.csv',parse_dates=['Year'])

# Rename columns
column_names = ['Country','Country Code','Year','Population']
df.columns = column_names

# Only years from 1800 onwards
df = df[df['Year'].astype(int) >= 1800]

# Convert Year column to datetime
df['Year'] = pd.to_datetime(df['Year'])

pivoted_df = df.pivot(index='Year',columns='Country',values='Population').fillna(0)
print(pivoted_df.head(5))

def current_total(values):
    total = values.sum()
    s = f'Total Population : {int(total):,}'
    return {'x': .85, 'y': .2, 's': s, 'ha': 'right', 'size': 11}

# Generate bar chart race
pivoted_df.plot_animated(filename='population-over-time-bar-chart-race.gif',
    n_visible=10,
    period_fmt="%Y",
    title='Top 10 Populous Countries 1800-2000',
    fixed_max=True,
    perpendicular_bar_func='mean',
    period_summary_func=current_total)

total_df = pivoted_df.sum(axis=1)
print(total_df)

total_df.plot_animated(kind='line',
    filename="total-population-over-time-line.gif",
    period_fmt="%Y",
    title="Total Population Over Time")
