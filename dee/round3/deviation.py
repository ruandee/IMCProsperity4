import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# --- 1. BLACK-SCHOLES CORE ---
def bs_call_price(S, K, T, r, sigma):
    if sigma <= 1e-6: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def solve_iv(market_price, S, K, T, r):
    if market_price <= max(0, S - K): return 0.0
    try:
        return brentq(lambda sigma: bs_call_price(S, K, T, r, sigma) - market_price, 1e-5, 4.0)
    except:
        return 0.0

# --- 2. THE 4-STEP IMPLEMENTATION ---
def process_data(file_paths, underlying_id='VELVETFRUIT_EXTRACT', T=1/250):
    all_data = []
    for f in file_paths:
        df = pd.read_csv(f, sep=';')
        
        # STEP 1: Fetch the mid_price of the underlying VEV at each timestamp
        underlying = df[df['product'] == underlying_id][['day', 'timestamp', 'mid_price']]
        underlying = underlying.rename(columns={'mid_price': 'S'})
        
        # Merge S into the dataset so every option row knows its current S
        df = df.merge(underlying, on=['day', 'timestamp'], how='left')
        
        # Filter for the VEV options (Strikes 5000-5500)
        options = df[df['product'].str.startswith('VEV_')].copy()
        options['strike'] = options['product'].str.extract('(\d+)').astype(float)
        options = options[(options['strike'] >= 5000) & (options['strike'] <= 5500)]
        
        all_data.append(options)
    
    df_merged = pd.concat(all_data)
    # Ensure sequential time for plotting
    df_merged['abs_tick'] = (df_merged['day'] * 1000000) + df_merged['timestamp']

    # STEP 2 & 3: Calculate IV with specific S and Moneyness (M = S/K)
    print("Calculating local IVs and Moneyness...")
    # Using rounding for caching speed (as discussed previously)
    df_merged['S_rounded'] = df_merged['S'].round(1)
    unique_states = df_merged[['mid_price', 'strike', 'S_rounded']].drop_duplicates()
    unique_states['iv'] = unique_states.apply(
        lambda row: solve_iv(row['mid_price'], row['S_rounded'], row['strike'], T, 0.0), axis=1
    )
    
    df = df_merged.merge(unique_states, on=['mid_price', 'strike', 'S_rounded'], how='left')
    df = df[df['iv'] > 0]
    df['moneyness'] = df['S'] / df['strike']

    # STEP 4: Fit the parabola to the (M, IV) pairs AT EACH TICK
    print("Fitting Parabolas and Detrending...")
    results = []
    for tick, group in df.groupby('abs_tick'):
        if len(group) < 3: continue
        
        # Fit parabola: IV = aM^2 + bM + c
        params = np.polyfit(group['moneyness'], group['iv'], 2)
        # Calculate deviation: Market IV - Parabola Predicted IV
        group['deviation'] = group['iv'] - np.polyval(params, group['moneyness'])
        results.append(group)

    return pd.concat(results)

# --- 3. PLOTTING ---
def plot_vouchers(df):
    plt.figure(figsize=(15, 7))
    vouchers = sorted(df['product'].unique(), key=lambda x: int(x.split('_')[1]))
    
    for v in vouchers:
        sub = df[df['product'] == v].sort_values('abs_tick')
        # iloc[::10] prevents the plot from being too heavy/laggy
        plt.plot(sub['abs_tick'].iloc[::10], sub['deviation'].iloc[::10], label=v, alpha=0.7)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("VEV IV Deviations: Adjusted for S, Sequential Time, and Parabolic Fit")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Run it
files = [
    'imcprosperity4/dee/round3/prices_round_3_day_0.csv', 
    'imcprosperity4/dee/round3/prices_round_3_day_1.csv',
    'imcprosperity4/dee/round3/prices_round_3_day_2.csv'
]
final_df = process_data(files)
plot_vouchers(final_df)