import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ---------------------------------------------------------
# 1. Load the Data from all days
# ---------------------------------------------------------
trades = pd.concat([
    pd.read_csv('imcprosperity4/dee/round3/trades_round_3_day_0.csv', sep=';'),
    pd.read_csv('imcprosperity4/dee/round3/trades_round_3_day_1.csv', sep=';'),
    pd.read_csv('imcprosperity4/dee/round3/trades_round_3_day_2.csv', sep=';')
], ignore_index=True)

prices = pd.concat([
    pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_0.csv', sep=';'),
    pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_1.csv', sep=';'),
    pd.read_csv('imcprosperity4/dee/round3/prices_round_3_day_2.csv', sep=';')
], ignore_index=True)

# ---------------------------------------------------------
# 2. Filter for products of interest
# ---------------------------------------------------------
# Exclude vouchers (VEV_*)
trades_filtered = trades[~trades['symbol'].str.startswith('VEV_')].copy()
trades_hp = trades_filtered[trades_filtered['symbol'] == 'HYDROGEL_PACK'].copy()
trades_vf = trades_filtered[trades_filtered['symbol'] == 'VELVETFRUIT_EXTRACT'].copy()

prices_hp = prices[prices['product'] == 'HYDROGEL_PACK'].copy()
prices_vf = prices[prices['product'] == 'VELVETFRUIT_EXTRACT'].copy()

# ---------------------------------------------------------
# 3. Basic Summary Statistics
# ---------------------------------------------------------
print("=" * 80)
print("HYDROGEL_PACK ANALYSIS")
print("=" * 80)
print(f"\nTotal trades: {len(trades_hp)}")
print(f"Date range: Day {trades_hp['timestamp'].min()//100000} to Day {trades_hp['timestamp'].max()//100000}")
print(f"\nPrice Statistics:")
print(trades_hp['price'].describe())
print(f"\nQuantity Statistics:")
print(trades_hp['quantity'].describe())
print(f"\nTotal volume traded: {trades_hp['quantity'].sum()}")

print("\n" + "=" * 80)
print("VELVETFRUIT_EXTRACT ANALYSIS")
print("=" * 80)
print(f"\nTotal trades: {len(trades_vf)}")
print(f"Date range: Day {trades_vf['timestamp'].min()//100000} to Day {trades_vf['timestamp'].max()//100000}")
print(f"\nPrice Statistics:")
print(trades_vf['price'].describe())
print(f"\nQuantity Statistics:")
print(trades_vf['quantity'].describe())
print(f"\nTotal volume traded: {trades_vf['quantity'].sum()}")

# ---------------------------------------------------------
# 4. Time Series Analysis
# ---------------------------------------------------------
# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# HYDROGEL_PACK - Price over time (all points)
ax = axes[0, 0]
ax.scatter(trades_hp['timestamp'], trades_hp['price'], alpha=0.6, s=50, color='steelblue')
ax.set_title('HYDROGEL_PACK: All Trade Prices Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Price')
ax.grid(True, alpha=0.3)

# VELVETFRUIT_EXTRACT - Price over time (all points)
ax = axes[0, 1]
ax.scatter(trades_vf['timestamp'], trades_vf['price'], alpha=0.6, s=50, color='coral')
ax.set_title('VELVETFRUIT_EXTRACT: All Trade Prices Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Price')
ax.grid(True, alpha=0.3)

# HYDROGEL_PACK - Volume over time
ax = axes[1, 0]
ax.scatter(trades_hp['timestamp'], trades_hp['quantity'], alpha=0.6, s=50, color='steelblue')
ax.set_title('HYDROGEL_PACK: Trade Quantities Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Quantity')
ax.grid(True, alpha=0.3)

# VELVETFRUIT_EXTRACT - Volume over time
ax = axes[1, 1]
ax.scatter(trades_vf['timestamp'], trades_vf['quantity'], alpha=0.6, s=50, color='coral')
ax.set_title('VELVETFRUIT_EXTRACT: Trade Quantities Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Quantity')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('products_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 5. Price Distribution Analysis
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# HYDROGEL_PACK - Price histogram
ax = axes[0, 0]
ax.hist(trades_hp['price'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('HYDROGEL_PACK: Price Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
ax.axvline(trades_hp['price'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades_hp['price'].mean():.2f}")
ax.axvline(trades_hp['price'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {trades_hp['price'].median():.2f}")
ax.legend()

# VELVETFRUIT_EXTRACT - Price histogram
ax = axes[0, 1]
ax.hist(trades_vf['price'], bins=50, color='coral', alpha=0.7, edgecolor='black')
ax.set_title('VELVETFRUIT_EXTRACT: Price Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
ax.axvline(trades_vf['price'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades_vf['price'].mean():.2f}")
ax.axvline(trades_vf['price'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {trades_vf['price'].median():.2f}")
ax.legend()

# HYDROGEL_PACK - Quantity distribution
ax = axes[1, 0]
ax.hist(trades_hp['quantity'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('HYDROGEL_PACK: Quantity Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Quantity')
ax.set_ylabel('Frequency')
ax.axvline(trades_hp['quantity'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades_hp['quantity'].mean():.2f}")
ax.legend()

# VELVETFRUIT_EXTRACT - Quantity distribution
ax = axes[1, 1]
ax.hist(trades_vf['quantity'], bins=30, color='coral', alpha=0.7, edgecolor='black')
ax.set_title('VELVETFRUIT_EXTRACT: Quantity Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Quantity')
ax.set_ylabel('Frequency')
ax.axvline(trades_vf['quantity'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {trades_vf['quantity'].mean():.2f}")
ax.legend()

plt.tight_layout()
plt.savefig('products_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 6. Bid-Ask Spread Analysis from Price Book Data
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# HYDROGEL_PACK - Bid-Ask spread over time
prices_hp_sorted = prices_hp.sort_values('timestamp')
prices_hp_sorted['spread'] = prices_hp_sorted['ask_price_1'] - prices_hp_sorted['bid_price_1']
ax = axes[0]
ax.scatter(prices_hp_sorted['timestamp'], prices_hp_sorted['spread'], alpha=0.6, s=30, color='steelblue')
ax.set_title('HYDROGEL_PACK: Bid-Ask Spread Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Spread (Ask - Bid)')
ax.grid(True, alpha=0.3)

# VELVETFRUIT_EXTRACT - Bid-Ask spread over time
prices_vf_sorted = prices_vf.sort_values('timestamp')
prices_vf_sorted['spread'] = prices_vf_sorted['ask_price_1'] - prices_vf_sorted['bid_price_1']
ax = axes[1]
ax.scatter(prices_vf_sorted['timestamp'], prices_vf_sorted['spread'], alpha=0.6, s=30, color='coral')
ax.set_title('VELVETFRUIT_EXTRACT: Bid-Ask Spread Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Spread (Ask - Bid)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('products_spreads.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 7. Price Volatility Analysis
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# HYDROGEL_PACK - Mid-price over time with all points
prices_hp_sorted = prices_hp.sort_values('timestamp').reset_index(drop=True)
ax = axes[0, 0]
ax.scatter(prices_hp_sorted['timestamp'], prices_hp_sorted['mid_price'], alpha=0.6, s=30, color='steelblue')
ax.plot(prices_hp_sorted['timestamp'], prices_hp_sorted['mid_price'], alpha=0.3, LineWidth=0.5, color='steelblue')
ax.set_title('HYDROGEL_PACK: Mid-Price Over Time (All Points)', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Mid-Price')
ax.grid(True, alpha=0.3)

# VELVETFRUIT_EXTRACT - Mid-price over time with all points
prices_vf_sorted = prices_vf.sort_values('timestamp').reset_index(drop=True)
ax = axes[0, 1]
ax.scatter(prices_vf_sorted['timestamp'], prices_vf_sorted['mid_price'], alpha=0.6, s=30, color='coral')
ax.plot(prices_vf_sorted['timestamp'], prices_vf_sorted['mid_price'], alpha=0.3, LineWidth=0.5, color='coral')
ax.set_title('VELVETFRUIT_EXTRACT: Mid-Price Over Time (All Points)', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Mid-Price')
ax.grid(True, alpha=0.3)

# HYDROGEL_PACK - Rolling volatility
prices_hp_sorted['returns'] = prices_hp_sorted['mid_price'].pct_change()
prices_hp_sorted['rolling_vol'] = prices_hp_sorted['returns'].rolling(window=20).std()
ax = axes[1, 0]
ax.scatter(prices_hp_sorted['timestamp'], prices_hp_sorted['rolling_vol'], alpha=0.6, s=30, color='steelblue')
ax.set_title('HYDROGEL_PACK: Rolling 20-Point Volatility', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Volatility (Std Dev of Returns)')
ax.grid(True, alpha=0.3)

# VELVETFRUIT_EXTRACT - Rolling volatility
prices_vf_sorted['returns'] = prices_vf_sorted['mid_price'].pct_change()
prices_vf_sorted['rolling_vol'] = prices_vf_sorted['returns'].rolling(window=20).std()
ax = axes[1, 1]
ax.scatter(prices_vf_sorted['timestamp'], prices_vf_sorted['rolling_vol'], alpha=0.6, s=30, color='coral')
ax.set_title('VELVETFRUIT_EXTRACT: Rolling 20-Point Volatility', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Volatility (Std Dev of Returns)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('products_volatility.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 8. Price vs Quantity Relationship
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# HYDROGEL_PACK - Price vs Quantity
ax = axes[0]
scatter = ax.scatter(trades_hp['price'], trades_hp['quantity'], alpha=0.6, s=50, 
                     c=trades_hp['timestamp'], cmap='viridis')
ax.set_title('HYDROGEL_PACK: Price vs Quantity (colored by time)', fontsize=12, fontweight='bold')
ax.set_xlabel('Price')
ax.set_ylabel('Quantity')
plt.colorbar(scatter, ax=ax, label='Timestamp')

# VELVETFRUIT_EXTRACT - Price vs Quantity
ax = axes[1]
scatter = ax.scatter(trades_vf['price'], trades_vf['quantity'], alpha=0.6, s=50,
                     c=trades_vf['timestamp'], cmap='plasma')
ax.set_title('VELVETFRUIT_EXTRACT: Price vs Quantity (colored by time)', fontsize=12, fontweight='bold')
ax.set_xlabel('Price')
ax.set_ylabel('Quantity')
plt.colorbar(scatter, ax=ax, label='Timestamp')

plt.tight_layout()
plt.savefig('products_price_vs_quantity.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 9. Trading Activity Over Time (by day)
# ---------------------------------------------------------
trades_hp['day'] = trades_hp['timestamp'] // 100000
trades_vf['day'] = trades_vf['timestamp'] // 100000

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# HYDROGEL_PACK - Trade count by day
ax = axes[0, 0]
trades_count_hp = trades_hp['day'].value_counts().sort_index()
ax.bar(trades_count_hp.index, trades_count_hp.values, color='steelblue', alpha=0.7)
ax.set_title('HYDROGEL_PACK: Number of Trades by Day', fontsize=12, fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Trade Count')

# VELVETFRUIT_EXTRACT - Trade count by day
ax = axes[0, 1]
trades_count_vf = trades_vf['day'].value_counts().sort_index()
ax.bar(trades_count_vf.index, trades_count_vf.values, color='coral', alpha=0.7)
ax.set_title('VELVETFRUIT_EXTRACT: Number of Trades by Day', fontsize=12, fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Trade Count')

# HYDROGEL_PACK - Total volume by day
ax = axes[1, 0]
volume_hp = trades_hp.groupby('day')['quantity'].sum()
ax.bar(volume_hp.index, volume_hp.values, color='steelblue', alpha=0.7)
ax.set_title('HYDROGEL_PACK: Total Volume by Day', fontsize=12, fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Total Quantity')

# VELVETFRUIT_EXTRACT - Total volume by day
ax = axes[1, 1]
volume_vf = trades_vf.groupby('day')['quantity'].sum()
ax.bar(volume_vf.index, volume_vf.values, color='coral', alpha=0.7)
ax.set_title('VELVETFRUIT_EXTRACT: Total Volume by Day', fontsize=12, fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Total Quantity')

plt.tight_layout()
plt.savefig('products_daily_activity.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 10. Advanced: Price Movement Analysis
# ---------------------------------------------------------
# Sort by timestamp to analyze price changes
trades_hp_sorted = trades_hp.sort_values('timestamp').reset_index(drop=True)
trades_vf_sorted = trades_vf.sort_values('timestamp').reset_index(drop=True)

trades_hp_sorted['price_change'] = trades_hp_sorted['price'].diff()
trades_vf_sorted['price_change'] = trades_vf_sorted['price'].diff()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# HYDROGEL_PACK - Price changes
ax = axes[0, 0]
ax.scatter(range(len(trades_hp_sorted)), trades_hp_sorted['price_change'], alpha=0.6, s=30, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_title('HYDROGEL_PACK: Price Changes Between Consecutive Trades', fontsize=12, fontweight='bold')
ax.set_xlabel('Trade Sequence')
ax.set_ylabel('Price Change')
ax.grid(True, alpha=0.3)

# VELVETFRUIT_EXTRACT - Price changes
ax = axes[0, 1]
ax.scatter(range(len(trades_vf_sorted)), trades_vf_sorted['price_change'], alpha=0.6, s=30, color='coral')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_title('VELVETFRUIT_EXTRACT: Price Changes Between Consecutive Trades', fontsize=12, fontweight='bold')
ax.set_xlabel('Trade Sequence')
ax.set_ylabel('Price Change')
ax.grid(True, alpha=0.3)

# HYDROGEL_PACK - Price change distribution
ax = axes[1, 0]
ax.hist(trades_hp_sorted['price_change'].dropna(), bins=40, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title('HYDROGEL_PACK: Distribution of Price Changes', fontsize=12, fontweight='bold')
ax.set_xlabel('Price Change')
ax.set_ylabel('Frequency')

# VELVETFRUIT_EXTRACT - Price change distribution
ax = axes[1, 1]
ax.hist(trades_vf_sorted['price_change'].dropna(), bins=40, color='coral', alpha=0.7, edgecolor='black')
ax.set_title('VELVETFRUIT_EXTRACT: Distribution of Price Changes', fontsize=12, fontweight='bold')
ax.set_xlabel('Price Change')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('products_price_changes.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 11. Summary Statistics for Price Changes
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("HYDROGEL_PACK - PRICE CHANGE ANALYSIS")
print("=" * 80)
print(trades_hp_sorted['price_change'].describe())
print(f"% of positive changes: {(trades_hp_sorted['price_change'] > 0).sum() / trades_hp_sorted['price_change'].notna().sum() * 100:.2f}%")

print("\n" + "=" * 80)
print("VELVETFRUIT_EXTRACT - PRICE CHANGE ANALYSIS")
print("=" * 80)
print(trades_vf_sorted['price_change'].describe())
print(f"% of positive changes: {(trades_vf_sorted['price_change'] > 0).sum() / trades_vf_sorted['price_change'].notna().sum() * 100:.2f}%")

# ---------------------------------------------------------
# 12. Mean Reversion Analysis
# ---------------------------------------------------------
def calculate_mean_reversion(price_series):
    mean_price = price_series.mean()
    deviations = []
    
    for i in range(len(price_series) - 1):
        current_price = price_series.iloc[i]
        deviation = current_price - mean_price
        
        if abs(deviation) > 0:
            ticks_to_revert = 0
            reverted = False
            
            for j in range(i + 1, min(i + 200, len(price_series))):
                future_price = price_series.iloc[j]
                future_deviation = future_price - mean_price
                ticks_to_revert += 1
                
                if (deviation > 0 and future_price <= mean_price) or (deviation < 0 and future_price >= mean_price):
                    reverted = True
                    break
            
            deviations.append({
                'index': i,
                'deviation': deviation,
                'abs_deviation': abs(deviation),
                'ticks_to_revert': ticks_to_revert,
                'reverted': reverted
            })
    
    return mean_price, pd.DataFrame(deviations)

print("\n" + "=" * 80)
print("MEAN REVERSION ANALYSIS")
print("=" * 80)

mean_hp, reversion_hp = calculate_mean_reversion(trades_hp_sorted['price'])
mean_vf, reversion_vf = calculate_mean_reversion(trades_vf_sorted['price'])

print("\nHYDROGEL_PACK:")
print(f"Mean Price: {mean_hp:.2f}")
print(f"Total price deviations tracked: {len(reversion_hp)}")
print(f"Prices that reverted to mean: {reversion_hp['reverted'].sum()} ({reversion_hp['reverted'].sum() / len(reversion_hp) * 100:.1f}%)")
print(f"\nMean reversion time (ticks):")
print(f"  Average: {reversion_hp['ticks_to_revert'].mean():.1f} ticks")
print(f"  Median: {reversion_hp['ticks_to_revert'].median():.0f} ticks")
print(f"  Min: {reversion_hp['ticks_to_revert'].min():.0f} ticks")
print(f"  Max: {reversion_hp['ticks_to_revert'].max():.0f} ticks")
print(f"  Std Dev: {reversion_hp['ticks_to_revert'].std():.1f} ticks")

print("\nVELVETFRUIT_EXTRACT:")
print(f"Mean Price: {mean_vf:.2f}")
print(f"Total price deviations tracked: {len(reversion_vf)}")
print(f"Prices that reverted to mean: {reversion_vf['reverted'].sum()} ({reversion_vf['reverted'].sum() / len(reversion_vf) * 100:.1f}%)")
print(f"\nMean reversion time (ticks):")
print(f"  Average: {reversion_vf['ticks_to_revert'].mean():.1f} ticks")
print(f"  Median: {reversion_vf['ticks_to_revert'].median():.0f} ticks")
print(f"  Min: {reversion_vf['ticks_to_revert'].min():.0f} ticks")
print(f"  Max: {reversion_vf['ticks_to_revert'].max():.0f} ticks")
print(f"  Std Dev: {reversion_vf['ticks_to_revert'].std():.1f} ticks")

# Plot mean reversion analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# HYDROGEL_PACK - Ticks to revert distribution
ax = axes[0, 0]
ax.hist(reversion_hp['ticks_to_revert'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(reversion_hp['ticks_to_revert'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f"Mean: {reversion_hp['ticks_to_revert'].mean():.1f}")
ax.set_title('HYDROGEL_PACK: Ticks to Mean Reversion', fontsize=12, fontweight='bold')
ax.set_xlabel('Ticks to Revert')
ax.set_ylabel('Frequency')
ax.legend()

# VELVETFRUIT_EXTRACT - Ticks to revert distribution
ax = axes[0, 1]
ax.hist(reversion_vf['ticks_to_revert'], bins=30, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(reversion_vf['ticks_to_revert'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean: {reversion_vf['ticks_to_revert'].mean():.1f}")
ax.set_title('VELVETFRUIT_EXTRACT: Ticks to Mean Reversion', fontsize=12, fontweight='bold')
ax.set_xlabel('Ticks to Revert')
ax.set_ylabel('Frequency')
ax.legend()

# HYDROGEL_PACK - Deviation vs Ticks to revert
ax = axes[1, 0]
scatter = ax.scatter(reversion_hp['abs_deviation'], reversion_hp['ticks_to_revert'], 
                    c=reversion_hp['reverted'], cmap='RdYlGn', alpha=0.6, s=50)
ax.set_title('HYDROGEL_PACK: Deviation vs Reversion Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Absolute Deviation from Mean')
ax.set_ylabel('Ticks to Revert')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Reverted')

# VELVETFRUIT_EXTRACT - Deviation vs Ticks to revert
ax = axes[1, 1]
scatter = ax.scatter(reversion_vf['abs_deviation'], reversion_vf['ticks_to_revert'],
                    c=reversion_vf['reverted'], cmap='RdYlGn', alpha=0.6, s=50)
ax.set_title('VELVETFRUIT_EXTRACT: Deviation vs Reversion Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Absolute Deviation from Mean')
ax.set_ylabel('Ticks to Revert')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Reverted')

plt.tight_layout()
plt.savefig('mean_reversion_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("Analysis complete! Plots saved as PNG files.")
print("=" * 80)
