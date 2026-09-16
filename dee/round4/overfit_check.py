"""
Overfitting diagnostics for the per-day smile fits.

Run this after your bootstrap has produced the per-day coefficient vectors.
Paste your actual per-day fits into the DAILY_FITS list below.

Format: (day_label, T, c0, c1, c2)
"""
import numpy as np
import math

# ─── Paste your per-day fits here ─────────────────────────────────────────
# Get these by adding a print(f"{path} T={T} beta={beta}") in bootstrap.py's
# main loop, or by re-running bootstrap with verbose output enabled below.
# Example format — replace with your real numbers:
DAILY_FITS = [
    # (label,          T,   c0,       c1,       c2      )
    ("r3_day0",        8,   1.21300800e-02,  -1.26856312e-02,  3.63124623e-01  ),
    ("r3_day1",        7,   1.18850685e-02,  +2.16013243e-02,  -4.62030195e-01),
    ("r3_day2",        6,   +1.18023291e-02,  +1.26533278e-02,  -1.03921020e-01),
    ("r3_day3",        5,   +7.34060952e-03  ,+1.64689130e-01,  -2.80429556e+00),
    ("r4_day1",        7,   1.18850685e-02,  +2.16013243e-02,  -4.62030195e-01),
    ("r4_day2",        6,   1.18023291e-02,  +1.26533278e-02,  -1.03921020e-01),
    ("r4_day3",        5,   7.34060952e-03,  +1.64689130e-01,  -2.80429556e+00),
]
# ──────────────────────────────────────────────────────────────────────────

M_STD  = 0.032102
F_REF  = 5246.87
WING   = 1.5 * M_STD
STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]

arr  = np.array([(r[2], r[3], r[4]) for r in DAILY_FITS])
TTEs = np.array([r[1] for r in DAILY_FITS])
labels = [r[0] for r in DAILY_FITS]
m_grid = np.array([math.log(K / F_REF) for K in STRIKES])


def smile(m, beta):
    return beta[0] + beta[1] * m + beta[2] * m * m


# ─── 1. Raw coefficient scatter ───────────────────────────────────────────
print("=" * 70)
print("1. Per-day coefficient values")
print("=" * 70)
print(f"{'Label':>12} | {'T':>4} | {'c0 (Level)':>12} | {'c1 (Skew)':>12} | {'c2 (Conv)':>12}")
print("-" * 60)
for (label, T, c0, c1, c2) in DAILY_FITS:
    print(f"{label:>12} | {T:>4} | {c0:>12.6f} | {c1:>12.6f} | {c2:>12.4f}")

mean = arr.mean(axis=0)
std  = arr.std(axis=0, ddof=1)
cv   = np.abs(std / (mean + 1e-12))
print("-" * 60)
print(f"{'mean':>12} |      | {mean[0]:>12.6f} | {mean[1]:>12.6f} | {mean[2]:>12.4f}")
print(f"{'std':>12} |      | {std[0]:>12.6f} | {std[1]:>12.6f} | {std[2]:>12.4f}")
print(f"{'CV (std/mean)':>12} |      | {cv[0]:>12.3f} | {cv[1]:>12.3f} | {cv[2]:>12.3f}")
print()
print("Rule of thumb: CV > 1 means std exceeds mean — coefficient is noise-dominated.")


# ─── 2. IV spread at each strike across days ──────────────────────────────
print()
print("=" * 70)
print("2. IV spread at each strike across days (this is what actually matters)")
print("   Large spread = fits disagree on the smile shape = overfit / noisy data")
print("=" * 70)
print(f"{'Strike':>8} | {'m':>7} | {'min IV':>7} | {'max IV':>7} | {'range':>7} | {'std':>7} | {'mean':>7}")
print("-" * 68)
for K, m in zip(STRIKES, m_grid):
    ivs = np.array([smile(m, beta) for beta in arr])
    clamped = "*" if abs(m) > WING else " "
    print(f"{K:>7}{clamped} | {m:>+7.4f} | {ivs.min()*100:>6.3f}% | "
          f"{ivs.max()*100:>6.3f}% | {(ivs.max()-ivs.min())*100:>6.3f}% | "
          f"{ivs.std(ddof=1)*100:>6.3f}% | {ivs.mean()*100:>6.3f}%")
print("  (* = outside wing clamp)")


# ─── 3. Leave-one-out cross-validation ───────────────────────────────────
print()
print("=" * 70)
print("3. Leave-one-out cross-validation (LOO-CV)")
print("   For each day: fit on other 6 days, predict this day's IV at each strike.")
print("   LOO error >> IV std → model is overfitting to noise rather than signal.")
print("=" * 70)
n = len(DAILY_FITS)
loo_errors = []
print(f"{'Label':>12} | {'T':>4} | {'mean |pred-obs| IV':>20} | {'max |pred-obs| IV':>20}")
print("-" * 64)
for i in range(n):
    train_idx = [j for j in range(n) if j != i]
    train_betas = arr[train_idx]
    mean_pred = train_betas.mean(axis=0)  # naive: predict with training mean

    obs_beta = arr[i]
    errors = []
    for m in m_grid:
        iv_pred = smile(m, mean_pred)
        iv_obs  = smile(m, obs_beta)
        errors.append(abs(iv_pred - iv_obs))
    mean_e = np.mean(errors) * 100
    max_e  = np.max(errors)  * 100
    loo_errors.append(mean_e)
    print(f"{labels[i]:>12} | {TTEs[i]:>4.0f} | {mean_e:>19.3f}% | {max_e:>19.3f}%")

print(f"\n  Mean LOO error across all days: {np.mean(loo_errors):.3f}%")
spread_ref = 0.5  # bid-ask half-spread typical
print(f"  For reference, typical half-spread ≈ {spread_ref/2*100:.3f}% IV units")
print(f"  If LOO error >> {spread_ref/2*100:.3f}%, you cannot reliably trade on smile residuals.")


# ─── 4. Same-TTE consistency check ───────────────────────────────────────
print()
print("=" * 70)
print("4. Same-TTE consistency (R3 vs R4 for same day)")
print("   If same market day gives very different smile shapes, data is noisy.")
print("=" * 70)
from collections import defaultdict
by_T = defaultdict(list)
for (label, T, c0, c1, c2) in DAILY_FITS:
    by_T[T].append((label, np.array([c0, c1, c2])))

for T in sorted(by_T):
    group = by_T[T]
    if len(group) < 2:
        continue
    print(f"  T={T}:")
    for label, beta in group:
        ivs_at_strikes = [smile(m, beta) * 100 for m in m_grid]
        print(f"    {label}: IVs = " + "  ".join(f"{iv:.3f}%" for iv in ivs_at_strikes))
    betas = np.array([b for _, b in group])
    mean_beta = betas.mean(axis=0)
    for label, beta in group:
        iv_diffs = [abs(smile(m, beta) - smile(m, mean_beta)) * 100 for m in m_grid]
        print(f"    {label} vs mean: max diff = {max(iv_diffs):.3f}%  mean diff = {np.mean(iv_diffs):.3f}%")
    print()


# ─── 5. R² of coefficient-vs-TTE linear trend ────────────────────────────
print("=" * 70)
print("5. R² of linear c_i(T) trend (used by linear extrapolation)")
print("   R² near 1 → trend is real. R² near 0 → no trend, just noise.")
print("   Low R² means option 5 extrapolation is unreliable.")
print("=" * 70)
A = np.column_stack([TTEs, np.ones_like(TTEs)])
for i, name in enumerate(["c0 (Level)", "c1 (Skew)", "c2 (Conv)"]):
    y = arr[:, i]
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    ss_res = float(((y - y_hat)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-20)
    slope, intercept = coef
    print(f"  {name}: slope={slope:+.4e}  R²={r2:.3f}", end="")
    if r2 < 0.5:
        print("  *** LOW — extrapolation unreliable ***", end="")
    print()