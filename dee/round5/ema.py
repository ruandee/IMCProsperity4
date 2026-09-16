import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Data & Create MultiIndex ─────────────────────────────────────────
file_path = "mightymerge.io__t5vy9kjz_pebbles.csv"
print(f"Loading and vectorizing {file_path}...")
df = pd.read_csv(file_path, sep=";")
df_pivot = df.pivot(index=['day', 'timestamp'], columns='product', values='mid_price')

# ── 2. Vectorized Beta & Spread Calculation ──────────────────────────────────
# We will hold the Beta window constant at 100 to isolate the Spread/Z-score optimization
beta_window = 100

# Calculate rolling variance of XS and covariance of XL, XS
var_xs = df_pivot['PEBBLES_XS'].rolling(window=beta_window).var()
cov_xl_xs = df_pivot['PEBBLES_XL'].rolling(window=beta_window).cov(df_pivot['PEBBLES_XS'])

# Rolling Beta
rolling_beta = cov_xl_xs / var_xs
rolling_beta = rolling_beta.fillna(1.0) # Fallback

# Calculate the stationary spread
spread = df_pivot['PEBBLES_XL'] - (rolling_beta * df_pivot['PEBBLES_XS'])

# ── 3. Define the Parameter Grid ─────────────────────────────────────────────
spread_windows = [20, 50, 100, 150, 200, 300]
entry_z_scores = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
exit_z = 0.5  # Held constant

results_matrix = np.zeros((len(spread_windows), len(entry_z_scores)))
results_matrix[:] = np.nan

print("Running vectorized parameter sweep...")

# ── 4. The Vectorized Engine ─────────────────────────────────────────────────
for i, window in enumerate(spread_windows):
    
    # Calculate Rolling Mean & Std for the Z-Score
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # Avoid division by zero
    spread_std = spread_std.replace(0, np.nan)
    z_score = (spread - spread_mean) / spread_std
    
    for j, entry_z in enumerate(entry_z_scores):
        
        # 1. Generate Signal Triggers
        # We want to be Long (+1) when z < -entry_z
        # We want to be Short (-1) when z > entry_z
        # We want to be Flat (0) when abs(z) < exit_z
        
        signals = pd.Series(np.nan, index=spread.index)
        signals[z_score < -entry_z] = 1
        signals[z_score > entry_z]  = -1
        signals[z_score.abs() < exit_z] = 0
        
        # 2. Forward Fill the State
        # If we enter a trade, we hold that state until the exit signal triggers a 0
        positions = signals.ffill().fillna(0)
        
        # 3. Calculate Vectorized PnL
        # We earn the difference in the spread, shifted by 1 to avoid lookahead bias
        spread_diff = spread.diff()
        pnl = positions.shift(1) * spread_diff
        
        # 4. Calculate pseudo-Sharpe (Mean / Std)
        valid_pnl = pnl.dropna()
        std_pnl = valid_pnl.std()
        
        if std_pnl > 0:
            sharpe = valid_pnl.mean() / std_pnl
            results_matrix[i, j] = sharpe
        else:
            results_matrix[i, j] = 0.0

# ── 5. Visualization ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.heatmap(
    results_matrix, 
    xticklabels=entry_z_scores, 
    yticklabels=spread_windows, 
    cmap="RdYlGn", 
    annot=True, 
    fmt=".4f"
)

plt.title("Z-Score Mean Reversion - Tick-Level Sharpe Ratio")
plt.xlabel("Entry Z-Score (Threshold)")
plt.ylabel("Spread Rolling Window (Ticks)")
plt.tight_layout()
plt.show()

# ── 6. Extract Best ──────────────────────────────────────────────────────────
best_idx = np.unravel_index(np.nanargmax(results_matrix), results_matrix.shape)
best_window = spread_windows[best_idx[0]]
best_z = entry_z_scores[best_idx[1]]
best_sharpe = results_matrix[best_idx]

print("\n--- OPTIMIZATION RESULTS ---")
print(f"Max Sharpe Ratio: {best_sharpe:.4f}")
print(f"Optimal Spread Window: {best_window}")
print(f"Optimal Entry Z-Score: {best_z}")