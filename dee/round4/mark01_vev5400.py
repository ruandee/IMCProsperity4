import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
PRICES_FILES = [f"imcprosperity4\dee/round4\data/round4\prices_round_4_day_{d}.csv" for d in [1, 2, 3]]
TRADES_FILES = [f"imcprosperity4\dee/round4\data/round4/trades_round_4_day_{d}.csv" for d in [1, 2, 3]]

TRADER   = "Mark 01"
PRODUCT  = "VEV_5400"
DAY_OFFSET     = 1_000_000
FORWARD_WINDOW = 500   # ticks ahead to measure price move

# ── LOAD ──────────────────────────────────────────────────────────────────────
price_frames, trade_frames = [], []

for day_idx, (pf, tf) in enumerate(zip(PRICES_FILES, TRADES_FILES)):
    offset = day_idx * DAY_OFFSET

    p = pd.read_csv(pf, sep=";")
    p["timestamp"] += offset
    price_frames.append(p)

    t = pd.read_csv(tf, sep=";")
    t["timestamp"] += offset
    trade_frames.append(t)

prices = pd.concat(price_frames, ignore_index=True).sort_values("timestamp")
trades = pd.concat(trade_frames, ignore_index=True).sort_values("timestamp")

# Filter to product
prod_prices = prices[prices["product"] == PRODUCT].sort_values("timestamp").reset_index(drop=True)
prod_trades = trades[trades["symbol"]  == PRODUCT].copy()

# Mark 01 trades: side = +1 buy, -1 sell
m_buys  = prod_trades[prod_trades["buyer"]  == TRADER].copy()
m_sells = prod_trades[prod_trades["seller"] == TRADER].copy()

m_buys["side"]   =  1
m_sells["side"]  = -1

m_trades = pd.concat([m_buys, m_sells]).sort_values("timestamp").reset_index(drop=True)

print("=" * 60)
print(f"  {TRADER}  |  {PRODUCT}")
print("=" * 60)

# ── 1. AVERAGE ORDER SIZE ──────────────────────────────────────────────────────
avg_size = m_trades["quantity"].mean()
avg_buy  = m_buys["quantity"].mean()
avg_sell = m_sells["quantity"].mean()

print(f"\n── Order Size ───────────────────────────────────────────")
print(f"  Total trades  : {len(m_trades)}  ({len(m_buys)} buys, {len(m_sells)} sells)")
print(f"  Avg size      : {avg_size:.2f}")
print(f"  Avg buy size  : {avg_buy:.2f}")
print(f"  Avg sell size : {avg_sell:.2f}")
print(f"  Min / Max     : {m_trades['quantity'].min()} / {m_trades['quantity'].max()}")

# ── 2. INTERVAL BETWEEN ORDERS ────────────────────────────────────────────────
m_trades["interval"] = m_trades["timestamp"].diff()

avg_interval    = m_trades["interval"].mean()
median_interval = m_trades["interval"].median()
min_interval    = m_trades["interval"].min()
max_interval    = m_trades["interval"].max()

print(f"\n── Order Intervals (ticks) ──────────────────────────────")
print(f"  Mean interval   : {avg_interval:.1f}")
print(f"  Median interval : {median_interval:.1f}")
print(f"  Min interval    : {min_interval:.1f}")
print(f"  Max interval    : {max_interval:.1f}")

# ── 3. PnL ────────────────────────────────────────────────────────────────────
# Cash flow: buys cost money (-), sells earn money (+)
# Remaining position valued at last mid price
m_trades["signed_qty"] = m_trades["side"] * m_trades["quantity"]
m_trades["cash_flow"]  = -m_trades["signed_qty"] * m_trades["price"]

realized_pnl   = 0.0
position       = 0
avg_cost       = 0.0
realized_flows = []

for _, row in m_trades.iterrows():
    qty  = int(row["signed_qty"])
    px   = row["price"]

    if qty > 0:  # buy
        total_cost = avg_cost * position + px * qty
        position  += qty
        avg_cost   = total_cost / position if position else 0
        realized_flows.append(0)
    else:        # sell
        sell_qty = abs(qty)
        realized_pnl += (px - avg_cost) * sell_qty
        position -= sell_qty
        if position < 0:
            avg_cost = px   # short position; reset
        realized_flows.append((px - avg_cost) * sell_qty)

last_mid      = prod_prices["mid_price"].iloc[-1]
unrealized_pnl = position * (last_mid - avg_cost) if position != 0 else 0
total_pnl     = realized_pnl + unrealized_pnl

print(f"\n── PnL ──────────────────────────────────────────────────")
print(f"  Realized PnL       : {realized_pnl:,.2f}")
print(f"  Remaining position : {position}  (avg cost {avg_cost:.4f})")
print(f"  Last mid price     : {last_mid:.4f}")
print(f"  Unrealized PnL     : {unrealized_pnl:,.2f}")
print(f"  Total PnL          : {total_pnl:,.2f}")

# ── 4. CORRELATION: order direction vs price move in next N ticks ─────────────
# For each trade, look up mid price at trade timestamp and at timestamp + FORWARD_WINDOW
# price_move = mid_price[t + FORWARD_WINDOW] - mid_price[t]
# side: +1 buy, -1 sell

price_ts  = prod_prices["timestamp"].values
price_mid = prod_prices["mid_price"].values

directions = []
fwd_moves  = []

for _, row in m_trades.iterrows():
    ts   = row["timestamp"]
    side = row["side"]

    # nearest index at or after ts
    idx_now = np.searchsorted(price_ts, ts, side="left")
    # index at ts + FORWARD_WINDOW
    idx_fwd = np.searchsorted(price_ts, ts + FORWARD_WINDOW, side="left")

    if idx_now >= len(price_ts) or idx_fwd >= len(price_ts):
        continue

    move = price_mid[idx_fwd] - price_mid[idx_now]
    directions.append(side)
    fwd_moves.append(move)

directions = np.array(directions)
fwd_moves  = np.array(fwd_moves)

if len(directions) > 1:
    corr = np.corrcoef(directions, fwd_moves)[0, 1]
    # fraction of buys followed by price rise / sells followed by price fall
    correct = np.mean(directions * fwd_moves > 0)
    avg_move_after_buy  = fwd_moves[directions ==  1].mean() if (directions ==  1).any() else float("nan")
    avg_move_after_sell = fwd_moves[directions == -1].mean() if (directions == -1).any() else float("nan")
else:
    corr = float("nan")
    correct = float("nan")
    avg_move_after_buy = avg_move_after_sell = float("nan")

print(f"\n── Order → Price Move Correlation (+{FORWARD_WINDOW} ticks) ─────────")
print(f"  Pearson correlation (side vs fwd move) : {corr:+.4f}")
print(f"  % trades where direction was 'right'   : {correct*100:.1f}%")
print(f"  Avg mid move after BUY                 : {avg_move_after_buy:+.4f}")
print(f"  Avg mid move after SELL                : {avg_move_after_sell:+.4f}")
print(f"  (positive buy corr = buys precede price rises)")

print("\n" + "=" * 60)