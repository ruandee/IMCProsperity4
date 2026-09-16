import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ==========================================
# 1. Option Pricing Math
# ==========================================
def black_scholes_call(S, K, T_days, r, sigma):
    if sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_days) / (sigma * np.sqrt(T_days))
    d2 = d1 - sigma * np.sqrt(T_days)
    return S * norm.cdf(d1) - K * np.exp(-r * T_days) * norm.cdf(d2)

def find_iv(price, S, K, T_days, r):
    if price <= max(0, S - K):
        return 0.0
    def objective(sigma):
        return black_scholes_call(S, K, T_days, r, sigma) - price
    try:
        # Constrain IV search between 0% and 500% (daily terms)
        return brentq(objective, 1e-5, 5.0)
    except (ValueError, RuntimeError):
        return 0.0

# ==========================================
# 2. Data Loading & Preparation
# ==========================================
files = [
    'imcprosperity4/dee/round3/prices_round_3_day_0.csv',
    'imcprosperity4/dee/round3/prices_round_3_day_1.csv',
    'imcprosperity4/dee/round3/prices_round_3_day_2.csv',
    'imcprosperity4/dee/round3/prices_round_3_day_3.csv',
]
df_list = []
for f in files:
    try:
        df_list.append(pd.read_csv(f, sep=";"))
    except FileNotFoundError:
        print(f"Warning: Could not find {f}")
df = pd.concat(df_list)

underlying = df[df["product"] == "VELVETFRUIT_EXTRACT"][["day", "timestamp", "mid_price"]].copy()
underlying = underlying.sort_values(["day", "timestamp"])
underlying = underlying.rename(columns={"mid_price": "S"})

# ==========================================
# 3. Filter Options & Calculate Metrics
# ==========================================
print("Mapping strikes and solving for daily IV...")
vouchers = df[df["product"].str.startswith("VEV_")].copy()
vouchers["K"] = vouchers["product"].str.extract(r"(\d+)").astype(float)

vouchers = vouchers[(vouchers["K"] >= 5000) & (vouchers["K"] <= 5500)]
data = pd.merge(vouchers, underlying[["day", "timestamp", "S"]], on=["day", "timestamp"])

r = 0.0

# Solve IV only on unique (mid_price, S, K, day) tuples — many timestamps
# share identical quotes, so this collapses the work substantially.
unique = data[["mid_price", "S", "K", "day"]].drop_duplicates().copy()

# TTE in days: day 0 -> 8, day 1 -> 7, day 2 -> 6, etc.
unique["T"] = 8 - unique["day"]

unique["iv"] = unique.apply(
    lambda row: find_iv(row["mid_price"], row["S"], row["K"], row["T"], r),
    axis=1,
)

data = pd.merge(data, unique, on=["mid_price", "S", "K", "day"])
data["moneyness"] = data["K"] / data["S"]

filtered = data[data["iv"] > 0.0001].copy()

# ==========================================
# 4. Unweighted Parabola Fit (in moneyness K/S)
# ==========================================
print("Fitting volatility smile (unweighted, M = K/S)...")

coeffs = np.polyfit(
    filtered["moneyness"],
    filtered["iv"],
    2,
)

print("\n" + "=" * 50)
print("FINAL VOLATILITY EQUATION (DAILY SCALED, UNWEIGHTED):")
print(f"IV = ({coeffs[0]:.6f} * M^2) + ({coeffs[1]:.6f} * M) + {coeffs[2]:.6f}")
print("     (Where M = Strike / Spot)")
print("=" * 50 + "\n")

# ==========================================
# 5. Plot
# ==========================================
plt.figure(figsize=(12, 7))
products = sorted(filtered["product"].unique())
colors = plt.cm.get_cmap("tab10", len(products))

for i, product_name in enumerate(products):
    subset = filtered[filtered["product"] == product_name]
    plt.scatter(
        subset["moneyness"], subset["iv"],
        label=product_name, color=colors(i), alpha=0.3, s=8,
    )

m_range = np.linspace(filtered["moneyness"].min(), filtered["moneyness"].max(), 200)
iv_trend = np.polyval(coeffs, m_range)
plt.plot(m_range, iv_trend, color='black', linewidth=3, label="Unweighted parabola fit")

plt.title("Volatility Smile (Daily Scaled, Unweighted)", fontsize=14)
plt.xlabel("Moneyness  (Strike / Spot)", fontsize=12)
plt.ylabel("Daily Implied Volatility", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()