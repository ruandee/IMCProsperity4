"""
Investigate the day3 (T=5) voucher market to understand why the smile fit
collapsed. Run this pointing at your actual day3 CSV.
"""
import math
import numpy as np
import pandas as pd

# ─── Edit ─────────────────────────────────────────────────────────────────
PATH       = "imcprosperity4/dee/round3/prices_round_3_day_3.csv"
UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKES    = [5000, 5100, 5200, 5300, 5400, 5500]
PREFIX     = "VEV_"
T          = 5.0       # TTE on this day
OU_MU      = 5246.87
SPREAD_FLOOR = 0.5
# ──────────────────────────────────────────────────────────────────────────

SQRT_2PI = math.sqrt(2.0 * math.pi)
def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _npdf(x): return math.exp(-0.5 * x * x) / SQRT_2PI
def bs_price(F, K, T, s):
    sq = s * math.sqrt(T); d1 = (math.log(F/K) + 0.5*s*s*T)/sq
    return F * _ncdf(d1) - K * _ncdf(d1 - sq)
def bs_vega(F, K, T, s):
    sq = s * math.sqrt(T); d1 = (math.log(F/K) + 0.5*s*s*T)/sq
    return F * _npdf(d1) * math.sqrt(T)
def implied_vol(P, F, K, T, s0=0.015):
    if P <= max(F-K, 0.0) + 1e-6: return None
    s = s0
    for _ in range(20):
        e = bs_price(F, K, T, s) - P
        if abs(e) < 1e-5: return s
        v = bs_vega(F, K, T, s)
        if v < 1e-10: break
        s = max(0.0005, min(0.3, s - e/v))
    return s if 0.001 < s < 0.3 else None

df = pd.read_csv(PATH, sep=";")
F = OU_MU  # forward ≈ MU given large theta

# ─── 1. How liquid is each voucher across the whole day? ──────────────────
print("=" * 70)
print("1. Voucher liquidity across the full day")
print("=" * 70)
print(f"{'Symbol':>12} | {'rows':>6} | {'bid nulls':>9} | {'ask nulls':>9} | "
      f"{'mean spread':>11} | {'max spread':>10} | {'mean mid':>9}")
print("-" * 80)
for K in STRIKES:
    sym = f"{PREFIX}{K}"
    sub = df[df["product"] == sym]
    n   = len(sub)
    bn  = sub["bid_price_1"].isna().sum()
    an  = sub["ask_price_1"].isna().sum()
    valid = sub.dropna(subset=["bid_price_1", "ask_price_1"])
    if len(valid):
        spreads = valid["ask_price_1"] - valid["bid_price_1"]
        print(f"{sym:>12} | {n:>6} | {bn:>9} | {an:>9} | "
              f"{spreads.mean():>11.2f} | {spreads.max():>10.2f} | "
              f"{valid['mid_price'].mean():>9.2f}")
    else:
        print(f"{sym:>12} | {n:>6} | {bn:>9} | {an:>9} | no valid quotes")

# ─── 2. Spot range on day3 vs other days ──────────────────────────────────
print()
print("=" * 70)
print("2. Underlying spot statistics")
print("=" * 70)
spot = df[df["product"] == UNDERLYING]["mid_price"].dropna()
print(f"  N ticks: {len(spot)}")
print(f"  Min: {spot.min():.2f}   Max: {spot.max():.2f}   "
      f"Mean: {spot.mean():.2f}   Std: {spot.std():.2f}")
print(f"  Range: {spot.max()-spot.min():.2f}")

# ─── 3. Sample IV at many timestamps across the day ──────────────────────
print()
print("=" * 70)
print("3. Implied vol at 20 evenly-spaced timestamps across the day")
print("   (shows whether the broken fit is a single-snapshot artifact)")
print("=" * 70)

# Build combined frame
out = {"S": df[df["product"] == UNDERLYING].set_index("timestamp")["mid_price"]}
for K in STRIKES:
    sub = df[df["product"] == f"{PREFIX}{K}"].set_index("timestamp")
    out[f"mid_{K}"] = sub["mid_price"]
    out[f"bid_{K}"] = sub["bid_price_1"]
    out[f"ask_{K}"] = sub["ask_price_1"]
combined = pd.DataFrame(out).dropna(how="any")
print(f"  Rows with complete quotes: {len(combined)} / "
      f"{len(df[df['product']==UNDERLYING])} underlying ticks")

if len(combined) == 0:
    print("  *** NO complete rows — vouchers and underlying never align in time ***")
    print("  This explains the broken fit: bootstrap had nothing to fit on.")
else:
    sample_idx = np.linspace(0, len(combined)-1, min(20, len(combined)), dtype=int)
    header = f"{'ts':>10} | {'S':>8} | {'F':>8} | " + \
             " | ".join(f"IV_{K:>4}" for K in STRIKES)
    print(header)
    print("-" * len(header))
    for i in sample_idx:
        row = combined.iloc[i]
        S = float(row["S"])
        ivs = []
        for K in STRIKES:
            mid = float(row[f"mid_{K}"])
            iv = implied_vol(mid, F, K, T)
            ivs.append(f"{iv*100:>6.3f}%" if iv else "  None ")
        ts = combined.index[i]
        print(f"{ts:>10} | {S:>8.2f} | {F:>8.2f} | " + " | ".join(ivs))

# ─── 4. Check intrinsic value violations ─────────────────────────────────
print()
print("=" * 70)
print("4. How often does mid_price fail the IV solver (intrinsic violation / OTM)?")
print("   If a strike's mid is at or below intrinsic, IV is undefined.")
print("=" * 70)
for K in STRIKES:
    sym = f"{PREFIX}{K}"
    sub = df[df["product"] == sym].dropna(subset=["mid_price"])
    if len(sub) == 0:
        print(f"  {sym}: no quotes")
        continue
    S_series = df[df["product"] == UNDERLYING].set_index("timestamp")["mid_price"]
    sub = sub.set_index("timestamp")
    aligned, S_aligned = sub["mid_price"].align(S_series)
    aligned = aligned.ffill()
    S_aligned = S_aligned.ffill()
    intrinsic = (S_aligned - K).clip(lower=0)
    violations = (sub["mid_price"] <= intrinsic + 1e-6).sum()
    pct = violations / len(sub) * 100
    flag = "  *** IV SOLVER WILL SKIP ***" if pct > 20 else ""
    print(f"  {sym}: {violations}/{len(sub)} ticks at/below intrinsic ({pct:.1f}%){flag}")

# ─── 5. WLS condition number at the mid-day snapshot ─────────────────────
print()
print("=" * 70)
print("5. WLS matrix condition number at mid-day snapshot")
print("   High condition (> 1e6) = near-singular fit = garbage coefficients")
print("=" * 70)
if len(combined) > 0:
    snap = combined.iloc[len(combined) // 2]
    S = float(snap["S"])
    K_arr, iv_arr, v_arr, sp_arr = [], [], [], []
    for K in STRIKES:
        mid = float(snap[f"mid_{K}"])
        bid = float(snap[f"bid_{K}"])
        ask = float(snap[f"ask_{K}"])
        spread = max(ask - bid, SPREAD_FLOOR)
        iv = implied_vol(mid, F, K, T)
        if iv is None:
            print(f"  K={K}: IV=None (skipped)")
            continue
        v = bs_vega(F, K, T, iv)
        K_arr.append(K); iv_arr.append(iv); v_arr.append(v); sp_arr.append(spread)
        print(f"  K={K}: IV={iv*100:.4f}%  vega={v:.4f}  spread={spread:.2f}  "
              f"weight={v/max(v_arr)*100:.1f}%")
    if len(K_arr) >= 3:
        K_np = np.array(K_arr, dtype=float)
        v_np = np.array(v_arr, dtype=float)
        sp_np = np.array(sp_arr, dtype=float)
        m_np = np.log(K_np / F)
        w_np = (v_np / v_np.max()) / sp_np
        X = np.column_stack([np.ones_like(m_np), m_np, m_np**2])
        Xw = X * w_np[:, None]
        A = Xw.T @ X
        cond = np.linalg.cond(A)
        print(f"\n  WLS matrix condition number: {cond:.3e}")
        if cond > 1e6:
            print("  *** NEAR-SINGULAR — coefficients are numerically unreliable ***")
        beta = np.linalg.lstsq(A, Xw.T @ np.array(iv_arr), rcond=None)[0]
        print(f"  Fitted coefficients: c0={beta[0]:.6f}  c1={beta[1]:.6f}  c2={beta[2]:.4f}")
        print(f"  Moneyness range used: [{m_np.min():+.4f}, {m_np.max():+.4f}]")
        print(f"  Span = {m_np.max()-m_np.min():.4f}  (need > ~0.05 for stable quadratic fit)")