import pandas as pd

files = ['imcprosperity4/dee/round3/prices_round_3_day_0.csv', 'imcprosperity4/dee/round3/prices_round_3_day_1.csv',\
          'imcprosperity4/dee/round3/prices_round_3_day_2.csv']
target_symbols = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
all_days_volume = []

for file in files:
    df = pd.read_csv(file, sep=";")
    df_filtered = df[df['symbol'].isin(target_symbols)]
    day_total = df_filtered.groupby('symbol')['quantity'].sum()
    all_days_volume.append(day_total)

combined = pd.concat(all_days_volume, axis=1)
combined = combined.fillna(0)
overall_avg = combined.mean(axis=1)

overall_avg_df = overall_avg.reset_index()
overall_avg_df.columns = ['Strike', 'Average_Executed_Volume']
overall_avg_df.to_csv("vev_average_executed_volume.csv", index=False)
print(overall_avg_df)