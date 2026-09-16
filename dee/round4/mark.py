import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
PRICES_FILES = [f"imcprosperity4\dee/round4\data/round4\prices_round_4_day_{d}.csv" for d in [1, 2, 3]]
TRADES_FILES = [f"imcprosperity4\dee/round4\data/round4/trades_round_4_day_{d}.csv" for d in [1, 2, 3]]
OUT_DIR      = "imcprosperity4\dee/round4\output"

TRADERS      = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]

DAY_OFFSET   = 1_000_000   # each day spans 0–999900; offset per day

BUY_COLOR    = "#3DDC84"
SELL_COLOR   = "#FF4D4D"
SPREAD_COLOR = "#4FC3F7"
MID_COLOR    = "#ECEFF4"
SPREAD_ALPHA = 0.18
BG_COLOR     = "#0D1117"
PANEL_COLOR  = "#161B22"
GRID_COLOR   = "#21262D"
TEXT_COLOR   = "#C9D1D9"
TRIANGLE_SIZE = 22

os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD & CONCATENATE ───────────────────────────────────────────────────────
print("Loading data ...")

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

products = sorted(prices["product"].unique())
print(f"Products ({len(products)}): {products}")

day_boundaries  = [d * DAY_OFFSET for d in range(1, 3)]
day_label_xs    = [d * DAY_OFFSET + DAY_OFFSET / 2 for d in range(3)]
day_labels      = ["Day 1", "Day 2", "Day 3"]

# ── PLOT ─────────────────────────────────────────────────────────────────────
for product in products:
    print(f"  Plotting {product} ...")

    prod_prices = prices[prices["product"] == product].sort_values("timestamp")
    prod_trades = trades[trades["symbol"]  == product]

    ts  = prod_prices["timestamp"].values
    mid = prod_prices["mid_price"].values
    bid = prod_prices["bid_price_1"].values.astype(float)
    ask = prod_prices["ask_price_1"].values.astype(float)

    fig, axes = plt.subplots(
        4, 2 , figsize=(24, 18), facecolor=BG_COLOR,
        gridspec_kw={"hspace": 0.50, "wspace": 0.22},
    )
    axes = axes.flatten()

    fig.suptitle(product, color=TEXT_COLOR, fontsize=20,
                 fontweight="bold", y=0.99, fontfamily="monospace")

    for idx, trader in enumerate(TRADERS):
        ax = axes[idx]
        ax.set_facecolor(PANEL_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7, zorder=0)

        # Spread band
        ax.fill_between(ts, bid, ask, color=SPREAD_COLOR, alpha=SPREAD_ALPHA,
                        linewidth=0, zorder=1)
        ax.plot(ts, bid, color=SPREAD_COLOR, linewidth=0.35, alpha=0.35, zorder=2)
        ax.plot(ts, ask, color=SPREAD_COLOR, linewidth=0.35, alpha=0.35, zorder=2)

        # Mid price
        ax.plot(ts, mid, color=MID_COLOR, linewidth=0.9, alpha=0.9, zorder=3, label="Mid")

        # Day dividers
        for bx in day_boundaries:
            ax.axvline(bx, color=TEXT_COLOR, linewidth=0.7, linestyle="--", alpha=0.35, zorder=4)

        # Buys & Sells
        buys  = prod_trades[prod_trades["buyer"]  == trader]
        sells = prod_trades[prod_trades["seller"] == trader]

        if not buys.empty:
            ax.scatter(buys["timestamp"], buys["price"],
                       marker="^", color=BUY_COLOR, s=TRIANGLE_SIZE,
                       linewidths=0, alpha=0.90, zorder=5,
                       label=f"Buy ({len(buys)})")

        if not sells.empty:
            ax.scatter(sells["timestamp"], sells["price"],
                       marker="v", color=SELL_COLOR, s=TRIANGLE_SIZE,
                       linewidths=0, alpha=0.90, zorder=5,
                       label=f"Sell ({len(sells)})")

        # X-axis: day labels at centre of each day
        ax.set_xticks(day_label_xs)
        ax.set_xticklabels(day_labels, fontsize=8.5, color=TEXT_COLOR)
        ax.set_xlim(ts.min(), ts.max())

        ax.set_title(trader, color=TEXT_COLOR, fontsize=11,
                     fontweight="bold", pad=6, fontfamily="monospace")
        ax.set_ylabel("Price", fontsize=7.5, color=TEXT_COLOR)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=6.5, loc="upper left",
                      framealpha=0.35, facecolor=BG_COLOR,
                      edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    out_path = os.path.join(OUT_DIR, f"{product}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    -> {out_path}")

print("\nDone.")