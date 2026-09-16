import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration based on your strategy context
PRODUCT = "ASH_COATED_OSMIUM"
EMA_ALPHA = 0.28  # [5]
SIGMA = 5.35      # [5]

# 1. Load and Concatenate Price Data (Direct Access Format)
df_p2 = pd.read_csv('prices_round_1_day_-2.csv', sep=';') # [2]
df_p1 = pd.read_csv('prices_round_1_day_-1.csv', sep=';') # [1]
df_p0 = pd.read_csv('prices_round_1_day_0.csv', sep=';')  # [3]

prices = pd.concat([df_p2, df_p1, df_p0])
prices = prices[prices['product'] == PRODUCT]
prices = prices.sort_values(by=['day', 'timestamp']).reset_index(drop=True)

# 2. Load and Concatenate Trade Data
df_t2 = pd.read_csv('trades_round_1_day_-2.csv', sep=';'); df_t2['day'] = -2 # [6]
df_t1 = pd.read_csv('trades_round_1_day_-1.csv', sep=';'); df_t1['day'] = -1 # [7]
df_t0 = pd.read_csv('trades_round_1_day_0.csv', sep=';'); df_t0['day'] = 0   # [8]

trades = pd.concat([df_t2, df_t1, df_t0])
trades = trades[trades['symbol'] == PRODUCT]
trades = trades.sort_values(by=['day', 'timestamp']).reset_index(drop=True)

# 3. Create Continuous Timeline (Handling 100k timestamp jump per day)
day_offsets = {-2: 0, -1: 100000, 0: 200000}
prices['time_seq'] = prices['timestamp'] + prices['day'].map(day_offsets)
trades['time_seq'] = trades['timestamp'] + trades['day'].map(day_offsets)

# 4. Calculate Diagnostic Indicators [5, 9]
prices['ema'] = prices['mid_price'].ewm(alpha=EMA_ALPHA, adjust=False).mean()
prices['imbalance'] = prices['bid_volume_1'] - prices['ask_volume_1']
prices['price_chg'] = prices['mid_price'].diff()

# 5. Diagnostic Dashboard Plotting
fig, axes = plt.subplots(5, 1, figsize=(15, 22), sharex=False)
plt.subplots_adjust(hspace=0.5)

# Panel 1: Price vs EMA (Lag Check)
axes.plot(prices['time_seq'], prices['mid_price'], label="Mid Price", color='royalblue', alpha=0.7)
axes.plot(prices['time_seq'], prices['ema'], label=f"EMA (α={EMA_ALPHA})", color='crimson', lw=1.5)
axes.set_title(f"{PRODUCT}: Price vs Fair Value (Lag Diagnostic)")
axes.legend()

# Panel 2: Market Executions (Execution Nodes)
axes[10].plot(prices['time_seq'], prices['mid_price'], color='lightgrey', alpha=0.5)
axes[10].scatter(trades['time_seq'], trades['price'], marker='x', color='purple', s=15, label="Actual Trades")
axes[10].set_title("Market Execution Nodes vs Mid Price")
axes[10].legend()

# Panel 3: Order Book Imbalance (Pressure Check)
axes[9].bar(prices['time_seq'], prices['imbalance'], color=['green' if x > 0 else 'red' for x in prices['imbalance']], alpha=0.6, width=100)
axes[9].set_title("Order Book Imbalance (Buying vs Selling Pressure)")

# Panel 4: Spread Dynamics (Shaded Area)
axes[11].fill_between(prices['time_seq'], prices['bid_price_1'], prices['ask_price_1'], color='grey', alpha=0.3, label="Spread")
axes[11].set_title("Level 1 Bid-Ask Spread Dynamics")

# Panel 5: Price Change Distribution vs Sigma Layers
axes[12].hist(prices['price_chg'].dropna(), bins=60, color='skyblue', edgecolor='black', alpha=0.7)
axes[12].axvline(0.2 * SIGMA, color='green', linestyle='--', label="Layer 1 Threshold (0.2σ)")
axes[12].axvline(-0.2 * SIGMA, color='green', linestyle='--')
axes[12].set_title(f"Price Change Distribution vs Strategy Layers (σ={SIGMA})")
axes[12].legend()

plt.show()