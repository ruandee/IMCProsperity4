"""
ccf_xs_xl.py
------------
Cross-Correlation Function analysis between PEBBLES_XS and PEBBLES_XL.
"""

import argparse
import json
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",     type=str, default=None, help="CSV price file")
parser.add_argument("--log",     type=str, default=None, help="Prosperity JSON log")
parser.add_argument("--max-lag", type=int, default=60,   help="Max CCF lag (default 60)")
parser.add_argument("--diff",    action="store_true",    help="Difference the series before CCF")
parser.add_argument("--day",     type=int, default=None, help="Filter to single day (1,2,3)")
args = parser.parse_args()

MAX_LAG = args.max_lag

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_csv(path):
    import csv
    ts, xs_mid, xl_mid = [], [], []
    # Using dictionaries to match timestamps for different product rows[cite: 1, 2]
    xs_data = {}
    xl_data = {}

    with open(path) as f:
        # Adjusted for your semicolon delimiter[cite: 1, 2]
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                t = float(row["timestamp"])
                product = row["product"]
                
                # Calculate mid price from bid_price_1 and ask_price_1[cite: 1, 2]
                mid = (float(row["bid_price_1"]) + float(row["ask_price_1"])) / 2
                
                if product == "PEBBLES_XS":
                    xs_data[t] = mid
                elif product == "PEBBLES_XL":
                    xl_data[t] = mid
            except (KeyError, ValueError):
                continue

    # Align the timestamps so we only compare times where both exist[cite: 1, 2]
    common_ts = sorted(list(set(xs_data.keys()) & set(xl_data.keys())))
    
    for t in common_ts:
        ts.append(t)
        xs_mid.append(xs_data[t])
        xl_mid.append(xl_data[t])
        
    return np.array(ts), np.array(xs_mid), np.array(xl_mid)

def load_log(path):
    """Parse a Prosperity sandbox/backtest JSON log."""
    with open(path) as f:
        raw = f.read()
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    ts, xs_mid, xl_mid = [], [], []
    for rec in records:
        state = rec.get("state", rec)
        t = state.get("timestamp", None)
        if t is None: continue
        od = state.get("order_depths", {})
        def mid_from_od(product):
            d = od.get(product, {})
            buys  = d.get("buy_orders",  {})
            sells = d.get("sell_orders", {})
            if isinstance(buys, dict):
                buys  = {int(k): v for k, v in buys.items()}
                sells = {int(k): v for k, v in sells.items()}
            best_bid = max(buys.keys())  if buys  else None
            best_ask = min(sells.keys()) if sells else None
            if best_bid and best_ask:
                return (best_bid + best_ask) / 2
            return None
        x  = mid_from_od("PEBBLES_XS")
        xl = mid_from_od("PEBBLES_XL")
        if x is not None and xl is not None:
            ts.append(float(t))
            xs_mid.append(x)
            xl_mid.append(xl)

    return np.array(ts), np.array(xs_mid), np.array(xl_mid)

def synthetic_data(n=2000):
    rng = np.random.default_rng(42)
    common = np.cumsum(rng.normal(0, 1, n))
    xl_noise = np.cumsum(rng.normal(0, 0.4, n))
    xs_noise = np.cumsum(rng.normal(0, 0.4, n))
    xl = 5000 + common + xl_noise
    xs = 4000 - common + xs_noise
    xs[3:] += 0.35 * xl[:-3] 
    ts = np.arange(n) * 100
    return ts, xs, xl

# ── Load data ──────────────────────────────────────────────────────────────────
if args.csv:
    print(f"Loading CSV: {args.csv}")
    timestamps, xs, xl = load_csv(args.csv)
elif args.log:
    print(f"Loading Prosperity log: {args.log}")
    timestamps, xs, xl = load_log(args.log)
else:
    print("No data file specified — using synthetic data.")
    timestamps, xs, xl = synthetic_data()

assert len(xs) == len(xl) == len(timestamps), "Series length mismatch"
n = len(xs)
print(f"Loaded {n} observations  (timestamps {timestamps[0]:.0f} → {timestamps[-1]:.0f})")

if args.day:
    print(f"Filtered to Day {args.day}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
if args.diff:
    xs = np.diff(xs);  xl = np.diff(xl);  timestamps = timestamps[1:]
    n = len(xs)
    print(f"First-differenced series (n={n})")

spread = xl - (-1.0143 * xs)   # raw spread with calibrated beta

# ── CCF Computation ───────────────────────────────────────────────────────────
def ccf(x, y, max_lag):
    """
    Returns lags and cross-correlations r(lag) = corr(x_t, y_{t+lag}).
    Positive lag = y leads x.  Negative lag = x leads y.
    """
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    N = len(x)
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag == 0:
            r = np.dot(x, y) / N
        elif lag > 0:
            r = np.dot(x[:-lag], y[lag:]) / (N - lag)
        else:
            r = np.dot(x[-lag:], y[:lag]) / (N + lag)
        corrs.append(r)
    return np.array(list(lags)), np.array(corrs)


def ewma(series, alpha):
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


# Main CCF: XS vs XL
lags, corrs_xs_xl = ccf(xs, xl, MAX_LAG)

# CCF on EWMA-smoothed series (alpha = 0.05 ≈ ~40-tick half-life)
ALPHA = 0.05
xs_ew = ewma(xs, ALPHA)
xl_ew = ewma(xl, ALPHA)
_, corrs_ew = ccf(xs_ew, xl_ew, MAX_LAG)

# CCF on spread (should be near zero if cointegrated)
_, corrs_spread = ccf(spread, spread, MAX_LAG)   # autocorrelation of spread
_, corrs_spread_xs = ccf(spread, xs, MAX_LAG)    # spread vs XS

# ── Statistics ────────────────────────────────────────────────────────────────
ci_95 = 1.96 / math.sqrt(n)   # Bartlett 95% confidence interval

peak_idx    = np.argmax(np.abs(corrs_xs_xl))
peak_lag    = lags[peak_idx]
peak_corr   = corrs_xs_xl[peak_idx]

# Contemporaneous correlation
contemp_idx  = np.where(lags == 0)[0][0]
contemp_corr = corrs_xs_xl[contemp_idx]

# Half-decay of correlation away from peak
def half_decay_lag(corrs, peak_idx, ci):
    """How many lags before correlation drops below CI from peak."""
    pk = abs(corrs[peak_idx])
    for offset in range(1, len(corrs)):
        for sign in [1, -1]:
            idx = peak_idx + sign * offset
            if 0 <= idx < len(corrs):
                if abs(corrs[idx]) < pk / 2:
                    return offset
    return MAX_LAG

decay = half_decay_lag(corrs_xs_xl, peak_idx, ci_95)

print("\n── CCF Summary ─────────────────────────────────────────────────────────")
print(f"  Observations        : {n}")
print(f"  95% CI band         : ±{ci_95:.4f}")
print(f"  Contemporaneous r   : {contemp_corr:>+.4f}")
print(f"  Peak |r|            : {peak_corr:>+.4f}  at lag={peak_lag:>+d}")
if peak_lag > 0:
    print(f"  → XL LEADS XS by {peak_lag} ticks  (XS lags)")
elif peak_lag < 0:
    print(f"  → XS LEADS XL by {-peak_lag} ticks  (XL lags)")
else:
    print(f"  → No lead/lag — contemporaneous")
print(f"  Correlation half-life: ~{decay} lags from peak")
print(f"  EWMA alpha={ALPHA} (half-life ≈ {math.log(0.5)/math.log(1-ALPHA):.0f} ticks)")
print(f"  Differenced          : {args.diff}")
print("────────────────────────────────────────────────────────────────────────")

# ── Plotting ──────────────────────────────────────────────────────────────────
DARK = "#0d1117";  PANEL = "#161b22";  BORDER = "#30363d"
TEXT = "#e6edf3";  DIM = "#8b949e"
C_RAW  = "#3b82f6"
C_EWMA = "#f97316"
C_SPR  = "#a855f7"
C_SPXS = "#22c55e"
C_CI   = "#ef4444"

fig = plt.figure(figsize=(16, 11), facecolor=DARK)
fig.suptitle("Cross-Correlation Function: PEBBLES_XS ↔ PEBBLES_XL",
             color=TEXT, fontsize=15, fontweight="bold", y=0.97)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.32,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)

def style(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=7)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, zorder=1)
    ax.axhline(0, color=DIM, linewidth=0.6)

def ci_bands(ax, lags, ci):
    ax.axhline( ci, color=C_CI, linewidth=1, linestyle="--", alpha=0.7, label="95% CI")
    ax.axhline(-ci, color=C_CI, linewidth=1, linestyle="--", alpha=0.7)
    ax.fill_between(lags,  ci, -ci, color=C_CI, alpha=0.05)

def stem_plot(ax, lags, corrs, color, label):
    pos = corrs >= 0
    ax.vlines(lags[pos],  0, corrs[pos],  color=color, linewidth=1.2, alpha=0.8)
    ax.vlines(lags[~pos], 0, corrs[~pos], color=color, linewidth=1.2, alpha=0.8)
    ax.scatter(lags, corrs, color=color, s=12, zorder=5, label=label)

# ── 1. Raw CCF ──
ax1 = fig.add_subplot(gs[0, :])
style(ax1, f"CCF: XS vs XL — raw mid prices  (lag>0 ⟹ XL leads XS)")
stem_plot(ax1, lags, corrs_xs_xl, C_RAW, "CCF(XS, XL)")
ci_bands(ax1, lags, ci_95)
if peak_lag != 0:
    ax1.axvline(peak_lag, color="#facc15", linewidth=1.2, linestyle=":", alpha=0.9,
                label=f"Peak lag={peak_lag:+d}  r={peak_corr:+.3f}")
ax1.axvline(0, color=DIM, linewidth=0.8, linestyle="-")
ax1.set_xlabel("Lag (ticks, positive = XL leads)", color=DIM, fontsize=8)
ax1.set_ylabel("Correlation", color=DIM, fontsize=8)
ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER, loc="upper right")
ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

# ── 2. EWMA-smoothed CCF ──
ax2 = fig.add_subplot(gs[1, 0])
style(ax2, f"CCF on EWMA-smoothed series (α={ALPHA})")
stem_plot(ax2, lags, corrs_ew, C_EWMA, f"CCF(EWMA_XS, EWMA_XL)")
ci_bands(ax2, lags, ci_95)
ax2.axvline(0, color=DIM, linewidth=0.8)
ax2.set_xlabel("Lag", color=DIM, fontsize=8)
ax2.set_ylabel("Correlation", color=DIM, fontsize=8)
ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)

# ── 3. Spread autocorrelation ──
ax3 = fig.add_subplot(gs[1, 1])
style(ax3, "ACF of Spread (xl + 1.0143·xs) — should decay fast if cointegrated")
stem_plot(ax3, lags, corrs_spread, C_SPR, "ACF(spread)")
ci_bands(ax3, lags, ci_95)
ax3.axvline(0, color=DIM, linewidth=0.8)
ax3.set_xlabel("Lag", color=DIM, fontsize=8)
ax3.set_ylabel("Correlation", color=DIM, fontsize=8)
ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)

# ── 4. Spread CCF vs XS ──
ax4 = fig.add_subplot(gs[2, 0])
style(ax4, "CCF: Spread vs XS — reveals which leg drives spread moves")
stem_plot(ax4, lags, corrs_spread_xs, C_SPXS, "CCF(spread, XS)")
ci_bands(ax4, lags, ci_95)
ax4.axvline(0, color=DIM, linewidth=0.8)
ax4.set_xlabel("Lag", color=DIM, fontsize=8)
ax4.set_ylabel("Correlation", color=DIM, fontsize=8)
ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)

# ── 5. Price series (for sanity check) ──
ax5 = fig.add_subplot(gs[2, 1])
style(ax5, "Price Series (first 500 obs)")
N_SHOW = min(500, n)
t_show = timestamps[:N_SHOW]
ax5.plot(t_show, xs[:N_SHOW],    color=C_RAW,  linewidth=0.9, label="XS")
ax5.plot(t_show, xl[:N_SHOW],    color=C_EWMA, linewidth=0.9, label="XL")
ax5r = ax5.twinx()
ax5r.plot(t_show, spread[:N_SHOW], color=C_SPR, linewidth=0.7, alpha=0.6, label="Spread")
ax5r.tick_params(colors=DIM, labelsize=7)
ax5r.set_ylabel("Spread", color=DIM, fontsize=7)
for sp in ax5r.spines.values(): sp.set_edgecolor(BORDER)
ax5.set_xlabel("Timestamp", color=DIM, fontsize=8)
ax5.set_ylabel("Mid Price", color=DIM, fontsize=8)
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5r.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2,
           fontsize=7, facecolor=PANEL, labelcolor=TEXT, edgecolor=BORDER)

# ── Annotation box ──
summary = (
    f"n={n}  |  CI=±{ci_95:.3f}  |  Peak lag={peak_lag:+d}  r={peak_corr:+.3f}\n"
    f"Contemp r={contemp_corr:+.3f}  |  Corr half-life≈{decay} lags"
)
fig.text(0.5, 0.005, summary, ha="center", va="bottom",
         color=DIM, fontsize=8,
         bbox=dict(facecolor=PANEL, edgecolor=BORDER, boxstyle="round,pad=0.3"))

out = "ccf_xs_xl.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nChart saved → {out}")
plt.show()