import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the historical price data
df_p1 = pd.read_csv('prices_round_0_day_-1.csv', sep=';')
df_p2 = pd.read_csv('prices_round_0_day_-2.csv', sep=';')
df = pd.concat([df_p2, df_p1])
df = df.sort_values(by=['day', 'timestamp']).reset_index(drop=True)

# 2. Filter for specific products
df_em = df[df['product'] == 'EMERALDS'].copy()
df_tom = df[df['product'] == 'TOMATOES'].copy()

# 3. Calculate spread width
df_em['spread'] = df_em['ask_price_1'] - df_em['bid_price_1']
df_tom['spread'] = df_tom['ask_price_1'] - df_tom['bid_price_1']

# --- PLOT 1: EMERALDS PRICE ACTION ---
fig, ax = plt.subplots(figsize=(12, 6))
sample_em = df_em.head(1000) # Look at the first 1000 ticks
ax.plot(sample_em['timestamp'], sample_em['ask_price_1'], label='Best Ask', color='red', alpha=0.7)
ax.plot(sample_em['timestamp'], sample_em['bid_price_1'], label='Best Bid', color='green', alpha=0.7)
ax.plot(sample_em['timestamp'], sample_em['mid_price'], label='Mid Price', color='blue', linestyle='--')
ax.set_title('EMERALDS')
ax.set_xlabel('Timestamp')
ax.set_ylabel('price')
ax.legend()
plt.tight_layout()
plt.show()

# --- PLOT 2: TOMATOES PRICE ACTION ---
fig, ax = plt.subplots(figsize=(12, 6))
sample_tom = df_tom.head(1000)
ax.plot(sample_tom['timestamp'], sample_tom['ask_price_1'], label='Best Ask', color='red', alpha=0.7)
ax.plot(sample_tom['timestamp'], sample_tom['bid_price_1'], label='Best Bid', color='green', alpha=0.7)
ax.plot(sample_tom['timestamp'], sample_tom['mid_price'], label='Mid Price', color='blue', linestyle='--')
ax.set_title('TOMATOES')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Price')
ax.legend()
plt.tight_layout()
plt.show()

# --- PLOT 3: SPREAD DISTRIBUTIONS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

spread_counts_em = df_em['spread'].value_counts().sort_index()
ax1.bar(spread_counts_em.index, spread_counts_em.values, color='grey', edgecolor='black')
ax1.set_title('Emeralds')
ax1.set_xlabel('Spread Width / tick')
ax1.set_ylabel('Freq')
ax1.set_xticks(spread_counts_em.index)

spread_counts_tom = df_tom['spread'].value_counts().sort_index()
ax2.bar(spread_counts_tom.index, spread_counts_tom.values, color='grey', edgecolor='black')
ax2.set_title('Tomatoes')
ax2.set_xlabel('Spread Width / tick')
ax2.set_ylabel('Freq')
ax2.set_xticks(spread_counts_tom.index)

plt.tight_layout()
plt.show()