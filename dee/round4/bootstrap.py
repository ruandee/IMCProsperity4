"""
Offline bootstrap. Processes 4 days of historical Prosperity price CSVs and
emits the seed constants for the live Kalman-filtered smile engine:

    X0           : (3,)  mean coefficient vector [Level, Skew, Convexity]
    P0           : (3,3) historical covariance across the 4 days
    M_STD        : sample std-dev of log-moneyness (wing-clamp boundary)
    OU_MU        : long-term mean of the underlying (AR(1) fit)
    OU_THETA     : mean-reversion speed, units of inverse-days

Pipeline per day:
    1. Pull underlying mid-price series and per-strike voucher quotes.
    2. Fit OU on the pooled spot history once (across all 4 days).
    3. At a representative snapshot per day, anchor on the OU forward,
       compute IVs, and run a WLS smile fit with weights
       w_i = (vega_i / max(vega)) / max(spread_i, SPREAD_FLOOR).
    4. Stack the 4 coefficient vectors → mean and sample covariance.

Edit PATHS below to point at your local Prosperity CSVs.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Edit me ──────────────────────────────────────────────────────────────
PATHS = [
    "imcprosperity4\dee/round3\prices_round_3_day_0.csv",
    "imcprosperity4\dee/round3\prices_round_3_day_1.csv",
    "imcprosperity4\dee/round3\prices_round_3_day_2.csv",
    "imcprosperity4\dee/round3\prices_round_3_day_3.csv",
]
UNDERLYING     = "VELVETFRUIT_EXTRACT"
STRIKES        = [5000, 5100, 5200, 5300, 5400, 5500]
VOUCHER_PREFIX = "VEV_"          # voucher symbol = f"{prefix}{K}"

TTE_START_DAYS = 5.0
TICKS_PER_DAY  = 999_000
SPREAD_FLOOR   = 0.5
VEGA_FLOOR     = 1e-8
# ──────────────────────────────────────────────────────────────────────────


SQRT_2PI = math.sqrt(2.0 * math.pi)


def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _npdf(x): return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_price(F, K, T, sig):
    sq = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / sq
    return F * _ncdf(d1) - K * _ncdf(d1 - sq)


def bs_vega(F, K, T, sig):
    sq = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / sq
    return F * _npdf(d1) * math.sqrt(T)


def implied_vol(P, F, K, T, s0=0.015):
    if P <= max(F - K, 0.0) + 1e-6:
        return None
    s = s0
    for _ in range(20):
        e = bs_price(F, K, T, s) - P
        if abs(e) < 1e-5:
            return s
        v = bs_vega(F, K, T, s)
        if v < 1e-10:
            break
        s = max(0.0005, min(0.3, s - e / v))
    return s if 0.001 < s < 0.3 else None


def load_day(path):
    """Returns dict: {'S': series, 'mid_K': series, 'bid_K': series, 'ask_K': series, ...}"""
    df = pd.read_csv(path, sep=";")
    out = {"S": df[df["product"] == UNDERLYING].set_index("timestamp")["mid_price"]}
    for K in STRIKES:
        sub = df[df["product"] == f"{VOUCHER_PREFIX}{K}"].set_index("timestamp")
        out[f"mid_{K}"] = sub["mid_price"]
        out[f"bid_{K}"] = sub["bid_price_1"]
        out[f"ask_{K}"] = sub["ask_price_1"]
    return pd.DataFrame(out)


def ou_fit(spot):
    """AR(1) on spot at the per-tick level. Returns (mu, theta_per_day)."""
    s = spot.dropna().values
    if len(s) < 50:
        raise ValueError("not enough spot ticks to fit OU")
    s0, s1 = s[:-1], s[1:]
    A = np.column_stack([s0, np.ones_like(s0)])
    a, b = np.linalg.lstsq(A, s1, rcond=None)[0]
    a = max(min(a, 0.999999), 0.5)
    theta_tick = -math.log(a)
    mu = b / (1.0 - a)
    # Convert from inverse-tick to inverse-day
    return float(mu), float(theta_tick * TICKS_PER_DAY)


def wls_one_day(snap, S, T, mu, theta):
    """Snap is a pd.Series row (one timestamp). Returns (beta, m_array) or None."""
    F = mu + (S - mu) * math.exp(-theta * T)

    K_arr, iv_arr, v_arr, spr_arr = [], [], [], []
    for K in STRIKES:
        mid = snap[f"mid_{K}"]
        bid = snap[f"bid_{K}"]
        ask = snap[f"ask_{K}"]
        if any(pd.isna([mid, bid, ask])):
            continue
        spread = max(ask - bid, SPREAD_FLOOR)
        iv = implied_vol(mid, F, K, T)
        if iv is None:
            continue
        v = max(bs_vega(F, K, T, iv), VEGA_FLOOR)
        K_arr.append(K); iv_arr.append(iv); v_arr.append(v); spr_arr.append(spread)

    if len(K_arr) < 4:
        return None

    K_np  = np.array(K_arr, dtype=float)
    iv_np = np.array(iv_arr, dtype=float)
    v_np  = np.array(v_arr, dtype=float)
    s_np  = np.array(spr_arr, dtype=float)

    m  = np.log(K_np / F)
    w  = (v_np / v_np.max()) / s_np
    X  = np.column_stack([np.ones_like(m), m, m * m])
    Xw = X * w[:, None]
    try:
        beta = np.linalg.solve(Xw.T @ X, Xw.T @ iv_np)
    except np.linalg.LinAlgError:
        return None
    return beta, m


def main():
    for p in PATHS:
        if not Path(p).exists():
            sys.exit(f"missing file: {p}")

    days = [load_day(p) for p in PATHS]

    # OU fit on pooled spot
    spot_all = pd.concat([d["S"] for d in days])
    mu, theta = ou_fit(spot_all)

    # Per-day WLS at a mid-day snapshot
    coeffs, m_pool = [], []
    for d_idx, df in enumerate(days):
        full = df.dropna(how="any")
        if full.empty:
            continue
        ts_mid = full.index[len(full) // 2]
        snap = full.loc[ts_mid]
        S = float(snap["S"])
        T = max(TTE_START_DAYS - d_idx, 1e-9)
        res = wls_one_day(snap, S, T, mu, theta)
        if res is None:
            continue
        beta, m = res
        coeffs.append(beta)
        m_pool.append(m)

    if len(coeffs) < 2:
        sys.exit("not enough valid daily fits")

    coeffs = np.array(coeffs)
    x0 = coeffs.mean(axis=0)
    P0 = np.cov(coeffs.T, ddof=1)
    m_std = float(np.std(np.concatenate(m_pool)))

    # Emit constants
    print("# === Auto-generated by bootstrap_constants.py ===")
    print("import numpy as np")
    print()
    print(f"X0    = np.array([{x0[0]:+.10e}, {x0[1]:+.10e}, {x0[2]:+.10e}])  "
          "# Level, Skew, Convexity")
    print("P0    = np.array([")
    for r in P0:
        print("    [" + ", ".join(f"{x:+.6e}" for x in r) + "],")
    print("])")
    print(f"M_STD     = {m_std:.6f}")
    print(f"OU_MU     = {mu:.4f}")
    print(f"OU_THETA  = {theta:.6e}  # per day")


if __name__ == "__main__":
    main()