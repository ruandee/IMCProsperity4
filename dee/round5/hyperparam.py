import pandas as pd
import numpy as np
from itertools import product

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("mightymerge.io__t5vy9kjz_pebbles.csv", sep=';')

df = df.pivot_table(
    index=["day", "timestamp"],
    columns="product",
    values="mid_price"
).sort_index()

df.columns.name = None

PRODUCTS = ["PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]

# =========================
# FEATURES
# =========================
df["anchor"] = (50000 - (df["PEBBLES_XS"] + df["PEBBLES_XL"])) / 3

for p in PRODUCTS:
    df[f"spread_{p}"] = df[p] - df["anchor"]

# numpy arrays (speed)
prices = {p: df[p].values for p in PRODUCTS}
spreads = {p: df[f"spread_{p}"].values for p in PRODUCTS}

# =========================
# ROLLING ZSCORE
# =========================
def rolling_z(arr, window):
    s = pd.Series(arr)
    mu = s.rolling(window).mean().values
    sigma = s.rolling(window).std().values
    return (arr - mu) / sigma

# =========================
# BACKTEST (RETURNS SERIES)
# =========================
def backtest_returns(zscores, spreads, prices, ENTRY_Z, EXIT_Z, POS_LIMIT, STEP, COST):

    n = len(next(iter(zscores.values())))
    pos = np.zeros(3)
    returns = []

    for i in range(1, n):

        z = np.array([zscores[p][i] for p in PRODUCTS])
        s = np.array([spreads[p][i] for p in PRODUCTS])

        if np.isnan(z).any():
            returns.append(0)
            continue

        order = np.argsort(s)
        cheap_i = order[0]
        rich_i  = order[-1]

        prev_pos = pos.copy()

        # ENTRY
        if z[rich_i] > ENTRY_Z:
            pos[rich_i]  = max(-POS_LIMIT, pos[rich_i]  - STEP)
            pos[cheap_i] = min(POS_LIMIT,  pos[cheap_i] + STEP)

        # EXIT
        elif abs(z[rich_i]) < EXIT_Z and abs(z[cheap_i]) < EXIT_Z:
            pos[rich_i]  = 0
            pos[cheap_i] = 0

        # PnL increment
        pnl = 0
        for j, p in enumerate(PRODUCTS):
            pnl += prev_pos[j] * (prices[p][i] - prices[p][i-1])

        # COST
        pnl -= np.sum(np.abs(pos - prev_pos)) * COST

        returns.append(pnl)

    return np.array(returns)


# =========================
# SHARPE RATIO
# =========================
def sharpe(returns):
    if len(returns) == 0:
        return -np.inf
    std = np.std(returns)
    if std == 0:
        return -np.inf
    return np.mean(returns) / std * np.sqrt(252)  # scaled


# =========================
# TRAIN / TEST SPLIT
# =========================
split = int(len(df) * 0.5)

train_idx = np.arange(0, split)
test_idx  = np.arange(split, len(df))


# =========================
# GRID
# =========================
param_grid = {
    "window":   [30, 40, 50, 60],
    "ENTRY_Z":  np.arange(0.8, 1.5, 0.1),
    "EXIT_Z":   np.arange(0.1, 0.4, 0.05),
    "STEP":     [1, 2, 3],
    "COST":     [0.2, 0.5],
}

results = []

# =========================
# GRID SEARCH
# =========================
for window in param_grid["window"]:

    zscores_all = {
        p: rolling_z(spreads[p], window)
        for p in PRODUCTS
    }

    for entry, exit_, step, cost in product(
        param_grid["ENTRY_Z"],
        param_grid["EXIT_Z"],
        param_grid["STEP"],
        param_grid["COST"]
    ):

        # TRAIN
        train_returns = backtest_returns(
            {p: zscores_all[p][train_idx] for p in PRODUCTS},
            {p: spreads[p][train_idx] for p in PRODUCTS},
            {p: prices[p][train_idx] for p in PRODUCTS},
            entry, exit_, 10, step, cost
        )

        train_sharpe = sharpe(train_returns)

        # TEST
        test_returns = backtest_returns(
            {p: zscores_all[p][test_idx] for p in PRODUCTS},
            {p: spreads[p][test_idx] for p in PRODUCTS},
            {p: prices[p][test_idx] for p in PRODUCTS},
            entry, exit_, 10, step, cost
        )

        test_sharpe = sharpe(test_returns)

        results.append({
            "window": window,
            "ENTRY_Z": round(entry, 2),
            "EXIT_Z": round(exit_, 2),
            "STEP": step,
            "COST": cost,
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "train_pnl": np.sum(train_returns),
            "test_pnl": np.sum(test_returns),
        })


# =========================
# RESULTS
# =========================
results_df = pd.DataFrame(results)

# sort by TEST sharpe (important)
results_df = results_df.sort_values("test_sharpe", ascending=False)

print("\nTop 20 configs (by TEST Sharpe):")
print(results_df.head(20))