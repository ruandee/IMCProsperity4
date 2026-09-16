import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import os

# ==========================================
# 1. Option Pricing & Greeks Math
# ==========================================
def black_scholes_call(S, K, T_days, r, sigma):
    if sigma <= 0 or T_days <= 0: return max(0.0, S - K)
    # Using T_days directly as per your daily-scaled bot logic
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_days) / (sigma * np.sqrt(T_days))
    d2 = d1 - sigma * np.sqrt(T_days)
    return S * norm.cdf(d1) - K * np.exp(-r * T_days) * norm.cdf(d2)

def find_iv(price, S, K, T_days, r):
    intrinsic = max(0, S - K)
    if price <= intrinsic: return 0.0001
    def objective(sigma):
        return black_scholes_call(S, K, T_days, r, sigma) - price
    try:
        return brentq(objective, 1e-5, 5.0)
    except:
        return 0.0001

def calc_vega(S, K, T_days, r, sigma):
    if sigma <= 0 or T_days <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_days) / (sigma * np.sqrt(T_days))
    return S * np.sqrt(T_days) * norm.pdf(d1)

files = ['imcprosperity4/dee/round3/prices_round_3_day_0.csv', 
         'imcprosperity4/dee/round3/prices_round_3_day_1.csv',
         'imcprosperity4/dee/round3/prices_round_3_day_2.csv']

daily_results = []
dte_values = [8, 7, 6]

for i, f in enumerate(files):
    if not os.path.exists(f): continue
    df = pd.read_csv(f, sep=";")
    
    # 1. Setup Underlying & Options
    underlying = df[df["product"] == "VELVETFRUIT_EXTRACT"].groupby("timestamp")["mid_price"].mean().to_frame("S")
    vouchers = df[df["product"].str.startswith("VEV_")].copy()
    vouchers["K"] = vouchers["product"].str.extract(r"(\d+)").astype(float)
    
    data = pd.merge(vouchers, underlying, on="timestamp")
    data["T"] = dte_values[i]
    
    # 2. Unique state caching for speed
    unique = data[["mid_price", "S", "K", "T"]].drop_duplicates().copy()
    unique["iv"] = unique.apply(lambda row: find_iv(row["mid_price"], row["S"], row["K"], row["T"], 0.0), axis=1)
    unique["vega"] = unique.apply(lambda row: calc_vega(row["S"], row["K"], row["T"], 0.0, row["iv"]), axis=1)
    
    # 3. Apply your Vega-Weighting
    fit_df = unique[unique["iv"] > 0.0001].copy()
    fit_df["M"] = fit_df["K"] / fit_df["S"] # YOUR MONEYNESS
    
    weights = np.sqrt(fit_df["vega"] / fit_df["vega"].max()) + 0.15
    
    # 4. Fit: IV = A*M^2 + B*M + C
    coeffs = np.polyfit(fit_df["M"], fit_df["iv"], 2, w=weights)
    daily_results.append({'dte': dte_values[i], 'A': coeffs[0], 'B': coeffs[1], 'C': coeffs[2]})

# --- EXTRAPOLATION TO TTE 5 ---
history = pd.DataFrame(daily_results)
prediction = {}

print("\n🚀 PREDICTED COEFFICIENTS FOR DAY 4 (TTE 5):")
for col in ['A', 'B', 'C']:
    line = np.polyfit(history['dte'], history[col], 1)
    prediction[col] = np.polyval(line, 5)
    print(f"   {col}: {prediction[col]:.8f}")

print(f"\nFORMULA: IV = ({prediction['A']:.6f} * M^2) + ({prediction['B']:.6f} * M) + {prediction['C']:.6f}")