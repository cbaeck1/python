import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import bar_chart_race as bcr

df = bcr.load_dataset('COVID-19')
# 누적합 :일자, 나라
df_sum = df.groupby(by=['date','countriesAndTerritories']).sum().groupby(level=[1]).cumsum()
df_sum = df_sum.reset_index(drop=False)
data_pivot = df_sum.pivot(index='date', columns='countriesAndTerritories', values='cases')

bcr.bar_chart_race(
        df=data_pivot, 
        filename='covid19_today.mp4', 
        orientation='h', 
        sort='desc', 
        n_bars=10, 
        fixed_order=False, 
        fixed_max=False, 
        steps_per_period=1, 
        period_length=360,
        #period_length=len(data_pivot.index) * len(data_pivot.columns), 
        end_period_pause=0,
        interpolate_period=False, 
        period_label={'x': .98, 'y': .3, 'ha': 'right', 'va': 'center', 'size':32}, 
        period_template='%B %d, %Y', 
        period_summary_func=lambda v, r: {'x': .98, 'y': .2, 
                                          's': f'Total deaths: {v.sum():,.0f}', 
                                          'ha': 'right', 'size':32}, 
        perpendicular_bar_func='median', 
        colors='dark12', 
        title=
        {
            'label': 'COVID-19 국가별 사망자',
            'size': 32,
            'color': 'Green',
            'weight': 'bold',
            'family': 'Malgun Gothic',
            'loc': 'center',
            'pad': 12
        },
        bar_size=.95, 
        bar_textposition='inside',
        bar_texttemplate='{x:,.0f}', 
        bar_label_font={'size':20, 'family':'Malgun Gothic', 'weight': 'bold', 'color':'Red'}, 
        tick_label_font={'size':20, 'family':'Malgun Gothic', 'weight': 'bold', 'color':'Blue'}, 
        tick_template= '{x:,.0f}', # ticker.StrMethodFormatter('{x:,.0f}')
        shared_fontdict={'size':20, 'family':'Malgun Gothic', 'color':'Black'},  
        scale='linear', 
        fig=None, 
        writer=None, 
        bar_kwargs={'alpha': .7},
        fig_kwargs={'figsize': (15, 8), 'dpi': 300},
        filter_column_colors=True,
        img_label_folder='country',
        tick_image_mode='trailing'
        ) 



