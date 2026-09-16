import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

def black_scholes_call(S, K, T, r, sigma):
    """Calculates Black-Scholes Call Price."""
    if T <= 1e-7 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2) * np.exp(-r * T)

def find_iv(price, S, K, T, r):
    """Solves for Implied Volatility."""
    if price <= max(0, S - K) + 0.01: # Intrinsic value check
        return 0.0
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    try:
        return newton(objective, 0.3, tol=1e-5, maxiter=100)
    except:
        return np.nan

# Parameters
days = [0, 1, 2]
strikes = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
r = 0.0
YEAR_TICKS = 250 * 1000000  # Scaling factor for T

all_iv_data = []
all_rv_data = []

for d in days:
    df = pd.read_csv(f'imcprosperity4/dee/round3/prices_round_3_day_{d}.csv', sep=';')
    
    # --- 1. Realized Volatility (RV) by Day ---
    df_und = df[df['product'] == 'VELVETFRUIT_EXTRACT'].sort_values('timestamp')
    log_returns = np.diff(np.log(df_und['mid_price']))
    # 10,000 intervals per day * 250 days/year
    rv_annualized = np.std(log_returns) * np.sqrt(10000 * 250)
    all_rv_data.append({'day': d, 'realized_volatility': rv_annualized})
    
    # --- 2. Implied Volatility (IV) by Strike & Day ---
    for k in strikes:
        df_opt = df[df['product'] == f'VEV_{k}'].sort_values('timestamp')
        merged = pd.merge(df_opt, df_und, on='timestamp', suffixes=('_opt', '_und'))
        
        ivs = []
        # Sample for speed (e.g., every 50th tick)
        for _, row in merged.iloc[::50].iterrows():
            # Time to expiry: end of Day 2
            rem_units = (2 - d) * 1000000 + (1000000 - row['timestamp'])
            T = max(rem_units / YEAR_TICKS, 1e-9)
            
            iv = find_iv(row['mid_price_opt'], row['mid_price_und'], k, T, r)
            if not np.isnan(iv) and iv > 0:
                ivs.append(iv)
        
        all_iv_data.append({
            'day': d, 
            'strike': k, 
            'avg_iv': np.mean(ivs) if ivs else np.nan
        })

# Output Results
rv_df = pd.DataFrame(all_rv_data)
iv_df = pd.DataFrame(all_iv_data).pivot(index='strike', columns='day', values='avg_iv')

print("Separated Realized Volatility (RV):")
print(rv_df)
print("\nImplied Volatility (IV) per Strike/Day:")
print(iv_df)

# Save results
rv_df.to_csv('daily_rv_results.csv', index=False)
iv_df.to_csv('strike_iv_results.csv')