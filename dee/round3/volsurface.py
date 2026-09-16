import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the dataset (Make sure all three files are in your directory)
df0 = pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_0.csv', sep=';')
df1 = pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_1.csv', sep=';')
df2 = pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_2.csv', sep=';')
df = pd.concat([df0, df1, df2])

# 2. Extract underlying asset (VELVETFRUIT_EXTRACT)
velvet = df[df['product'] == 'VELVETFRUIT_EXTRACT'][['day', 'timestamp', 'mid_price']]
velvet.rename(columns={'mid_price': 'S'}, inplace=True)

# 3. Extract European options (VEV_*)
options = df[df['product'].str.startswith('VEV_')].copy()
options['K'] = options['product'].str.split('_').str[1].astype(float)
options.rename(columns={'mid_price': 'C'}, inplace=True)

# 4. Merge data so underlying price maps accurately to the options timestamp
merged = pd.merge(options[['day', 'timestamp', 'product', 'K', 'C']], velvet, on=['day', 'timestamp'])

# 5. Define Time To Expiry (TTE)
# Your manual notes that Day 0 is 7 days to expiry. We decrement to 6 and 5.
merged['T_days'] = 7 - merged['day']
merged['T_years'] = merged['T_days'] / 252.0 # Annualize assuming 252 trading days

# 6. Calculate Moneyness (K / S)
# < 1 means In-The-Money (for calls), = 1 is At-The-Money, > 1 is Out-Of-The-Money
merged['Moneyness'] = merged['K'] / merged['S']

# 7. Define Black-Scholes Formula and Brent Root Finder
def bs_call(S, K, T, r, sigma):
    """Calculates the theoretical price of a European Call"""
    if T <= 0 or sigma <= 0: return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol(price, S, K, T, r=0.0):
    """Back-solves for Implied Volatility"""
    if price <= np.maximum(S - K, 0): return 1e-6 # Drop intrinsic value boundary cases
    def obj(sigma): return bs_call(S, K, T, r, sigma) - price
    try:
        return brentq(obj, 1e-6, 5.0)
    except ValueError:
        return np.nan

# 8. Filter and Calculate Volatility
# Using subset logic to bypass processing 3M+ rows locally which would freeze scripts. 
sample_timestamps = merged['timestamp'].unique()[::200]
sample = merged[merged['timestamp'].isin(sample_timestamps)].copy()

print("Calculating IV...")
sample['IV'] = sample.apply(lambda row: implied_vol(row['C'], row['S'], row['K'], row['T_years']), axis=1)
sample.dropna(subset=['IV'], inplace=True)

# 9. Plot the 3D Surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Creating a 3D scatter plot 
sc = ax.scatter(sample['Moneyness'], sample['T_days'], sample['IV'], 
                c=sample['IV'], cmap='viridis', marker='o', s=15, alpha=0.8)

ax.set_xlabel('\nMoneyness (Strike/Spot)', linespacing=3.2)
ax.set_ylabel('\nTime to Expiry (Days)', linespacing=3.2)
ax.set_zlabel('\nImplied Volatility', linespacing=3.2)
ax.set_title('Implied Volatility Surface: Moneyness vs TTE')
fig.colorbar(sc, ax=ax, label='Implied Volatility', pad=0.1)

# Rotate to best reveal the Volatility Skew
ax.view_init(elev=20, azim=135)

plt.tight_layout()
plt.show()