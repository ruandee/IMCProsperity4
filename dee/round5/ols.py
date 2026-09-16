import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ── 1. Load Data ─────────────────────────────────────────────────────────────
file_path = "mightymerge.io__t5vy9kjz_pebbles.csv"
df = pd.read_csv(file_path, sep=";")
df_pivot = df.pivot(index=['day', 'timestamp'], columns='product', values='mid_price')

# ── 2. OLS Regression to find optimal Static Beta ────────────────────────────
# We are regressing XL (Dependent) against XS (Independent)
# Equation: XL = (Beta * XS) + Alpha (Constant/Offset)

Y = df_pivot['PEBBLES_XL'].dropna()
X = df_pivot['PEBBLES_XS'].dropna()

# Add a constant to the independent variable to calculate the alpha offset
X_sm = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(Y, X_sm).fit()

optimal_beta = model.params['PEBBLES_XS']
optimal_alpha = model.params['const']

print("--- OLS REGRESSION RESULTS ---")
print(f"Optimal Static Beta (Hedge Ratio): {optimal_beta:.4f}")
print(f"Optimal Alpha (Offset): {optimal_alpha:.4f}")
print(f"R-Squared (Correlation Strength): {model.rsquared:.4f}")

# ── 3. Visualize the Stationary Spread ───────────────────────────────────────
# Calculate the spread using our new optimal mathematical constants
stationary_spread = Y - (optimal_beta * X) - optimal_alpha

plt.figure(figsize=(12, 5))
plt.plot(stationary_spread.values, label='Stationary Spread (Residuals)')
plt.axhline(0, color='red', linestyle='--', label='Mean (0)')
plt.title(f"Cointegrated Spread (Static Beta: {optimal_beta:.2f})")
plt.legend()
plt.show()