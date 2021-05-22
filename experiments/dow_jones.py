import matplotlib.pyplot as plt
import pandas as pd

# Format column datatypes
dow_df = pd.read_csv('../data/dow_jones_index.data')
'''dow_df['open'] = dow_df['open'].str.replace('$', '').astype(float)
dow_df['high'] = dow_df['high'].str.replace('$', '').astype(float)
dow_df['low'] = dow_df['low'].str.replace('$', '').astype(float)
dow_df['close'] = dow_df['close'].str.replace('$', '').astype(float)
dow_df['next_weeks_open'] = dow_df['next_weeks_open'].str.replace('$', '').astype(float)
dow_df['next_weeks_close'] = dow_df['next_weeks_close'].str.replace('$', '').astype(float)'''
dow_df['date'] = pd.to_datetime(dow_df['date'])

fig = plt.figure(figsize=(15, 15))
for stock in dow_df['stock'].unique():
    plt.plot(dow_df['date'].unique(), dow_df[dow_df['stock'] == stock].groupby('date')['open'].sum())
plt.legend(dow_df['stock'].unique())
plt.show()
