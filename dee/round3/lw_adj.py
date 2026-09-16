import pandas as pd
import numpy as np
import math

# Load the datasets
df0 = pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_0.csv', sep=';')
df1 = pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_1.csv', sep=';')
df2 = pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_2.csv', sep=';')

# Combine them and filter for VELVETFRUIT_EXTRACT
df = pd.concat([df0, df1, df2], ignore_index=True)
df_vf = df[df['product'] == 'VELVETFRUIT_EXTRACT'].copy()

# Sort chronologically by day and timestamp
df_vf.sort_values(by=['day', 'timestamp'], inplace=True)
prices = df_vf['mid_price'].dropna()

# 1. Calculate continuously compounded returns: r = log(P_t / P_{t-1})
returns = np.log(prices / prices.shift(1)).dropna()

# 2. Estimate the first-order autocorrelation of the returns: rho(1)
rho_1 = returns.autocorr(lag=1)

# Time to Expiry (tau_years) - Day 3 is 4 days to expiry
tau_years = 4.0 / 252.0

# 3. Solve for the speed of adjustment (gamma) based on the O-U specification:
gamma = - (1.0 / tau_years) * math.log(1.0 + 2.0 * rho_1)
        
# 4. Compute the actual adjustment factor
lw_adj = (gamma * tau_years) / (1.0 - math.exp(-gamma * tau_years))

print(f"Number of price data points: {len(prices)}")
print(f"First-order autocorrelation (rho_1): {rho_1:.6f}")
print(f"Gamma: {gamma:.6f}")
print(f"Lo-Wang Adjustment Factor: {lw_adj:.6f}")