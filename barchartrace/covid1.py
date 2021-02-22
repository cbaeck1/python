import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import bar_chart_race as bcr


df = bcr.load_dataset('covid19_tutorial')
# df = bcr.load_dataset('covid19')
print(df.info())
print(df)

# current_year = 2018
# dff = (df[df['year'].eq(current_year)]
#        .sort_values(by='value', ascending=True)
#        .head(10))
# print(dff)

# fig, ax = plt.subplots(figsize=(15, 8))
# ax.barh(dff['name'], dff['value'])
# plt.show()


bcr.bar_chart_race(
        df=df, 
        filename='covid19_deaths.mp4', 
        orientation='h', 
        sort='desc', 
        n_bars=12, 
        fixed_order=False, 
        fixed_max=False, 
        steps_per_period=10, 
        period_length=1000, 
        end_period_pause=0,
        interpolate_period=False, 
        period_label={'x': .98, 'y': .3, 'ha': 'right', 'va': 'center', 'size':32}, 
        period_template='%B %d, %Y', 
        period_summary_func=lambda v, r: {'x': .98, 'y': .2, 
                                          's': f'Total deaths: {v.sum():,.0f}', 
                                          'ha': 'right', 
                                          'weight': 'bold',
                                          'size':32}, 
        perpendicular_bar_func='median', 
        colors='dark12_r', 
        title=
        {
            'label': 'COVID-19 국가별 사망자',
            'size': 32,
            'color': 'Green',
            'weight': 'bold',
            'family': 'NanumGothic',
            'loc': 'center',
            'pad': 12
        },
        bar_size=.95, 
        bar_textposition='inside',
        bar_texttemplate='{x:,.0f}', 
        bar_label_font={'size':12, 'family':'NanumGothic', 'weight': 'bold', 'color':'White'}, 
        tick_label_font={'size':20, 'family':'NanumGothic', 'weight': 'bold', 'color':'Blue'}, 
        tick_template= '{x:,.0f}', # ticker.StrMethodFormatter('{x:,.0f}')
        shared_fontdict={'size':20, 'family':'NanumGothic', 'color':'Black'},  
        scale='linear', 
        fig=None, 
        writer=None, 
        bar_kwargs={'alpha': .6},
        fig_kwargs={'figsize': (15, 8), 'dpi': 500},
        filter_column_colors=True,
        img_label_folder='country',
        tick_label_mode='mixed',
        tick_image_mode='trailing'
        ) 



