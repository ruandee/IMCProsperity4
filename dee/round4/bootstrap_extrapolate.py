"""
Offline bootstrap. Processes historical Prosperity price CSVs from rounds 3
and 4 and emits the seed constants for the live Kalman-filtered smile engine:

    X0           : (3,)  mean coefficient vector [Level, Skew, Convexity]
    P0           : (3,3) historical covariance across the daily fits
    M_STD        : sample std-dev of log-moneyness (wing-clamp boundary)
    OU_MU        : long-term mean of the underlying (AR(1) fit)
    OU_THETA     : mean-reversion speed, units of inverse-days

Round-3 day-N and Round-4 day-N share the same `day_idx` (and therefore the
same TTE), since they represent the same market conditions. Each file still
contributes its own WLS coefficient vector, giving a richer P0.

Pipeline:
    1. Pull underlying mid-price series and per-strike voucher quotes from
       every file.
    2. Fit OU on the pooled spot history once (across all files).
    3. At a representative snapshot per file, anchor on the OU forward,
       compute IVs, and run a WLS smile fit with weights
       w_i = (vega_i / max(vega)) / max(spread_i, SPREAD_FLOOR).
    4. Stack the resulting coefficient vectors → mean and sample covariance.

Edit PATHS below to point at your local Prosperity CSVs.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Edit me ──────────────────────────────────────────────────────────────
# (path, day_idx) pairs. `day_idx` sets the TTE: T = TTE_START_DAYS - day_idx.
# Round-3 and Round-4 files for the same calendar day share day_idx.
PATHS = [
    ("imcprosperity4/dee/round3/prices_round_3_day_0.csv", 0),
    ("imcprosperity4/dee/round3/prices_round_3_day_1.csv", 1),
    ("imcprosperity4/dee/round3/prices_round_3_day_2.csv", 2),
    ("imcprosperity4/dee/round3/prices_round_3_day_3.csv", 3),
    ("imcprosperity4\dee/round4\data/round4\prices_round_4_day_1.csv", 1),
    ("imcprosperity4\dee/round4\data/round4\prices_round_4_day_1.csv", 2),
    ("imcprosperity4\dee/round4\data/round4\prices_round_4_day_1.csv", 3),
]
UNDERLYING     = "VELVETFRUIT_EXTRACT"
STRIKES        = [5000, 5100, 5200, 5300, 5400, 5500]
VOUCHER_PREFIX = "VEV_"          # voucher symbol = f"{prefix}{K}"

TTE_START_DAYS = 8.0             # TTE in days at day_idx == 0
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
    """Returns DataFrame with columns S, mid_K, bid_K, ask_K for each strike."""
    df = pd.read_csv(path, sep=";")
    out = {"S": df[df["product"] == UNDERLYING].set_index("timestamp")["mid_price"]}
    for K in STRIKES:
        sub = df[df["product"] == f"{VOUCHER_PREFIX}{K}"].set_index("timestamp")
        out[f"mid_{K}"] = sub["mid_price"]
        out[f"bid_{K}"] = sub["bid_price_1"]
        out[f"ask_{K}"] = sub["ask_price_1"]
    return pd.DataFrame(out)


def ou_fit(spot_series_list,
           sample_intervals=(100, 1000, 5000, 25000, 100_000, 250_000, 500_000)):
    """Fit AR(1) at multiple timescales to separate microstructure noise
    from real mean reversion.

    Microstructure (bid-ask bounce, quote flicker) creates spurious fast
    mean-reversion at tick granularity. As the sampling interval grows past
    the microstructure timescale, theta_per_day stabilizes — that plateau
    is the real OU rate relevant for multi-day option pricing.

    For each interval `dt_target` (in timestamp units), we subsample each
    day's spot series to that frequency (greedy walk), fit AR(1) on the
    pairs (s_t, s_{t+dt}), and convert to theta_per_day.

    Selection rule:
        Pick the longest interval that (a) has >= MIN_PAIRS pairs and
        (b) shows reasonable convergence vs. the previous timescale
        (|theta_new - theta_prev| / theta_new < CONV_TOL). If nothing
        satisfies (b), fall back to the longest interval with enough data
        and FLAG the result as non-converged.
    """
    MIN_PAIRS = 12
    CONV_TOL  = 0.5     # theta within 50% of the previous timescale's

    def fit_one(dt_target):
        s0_list, s1_list = [], []
        for spot in spot_series_list:
            s = spot.dropna().sort_index()
            if len(s) < 5:
                continue
            ts = s.index.values.astype(float)
            vals = s.values.astype(float)
            i = 0
            anchors = [0]
            while i < len(ts) - 1:
                target = ts[i] + dt_target
                j = i + 1
                while j < len(ts) and ts[j] < target:
                    j += 1
                if j >= len(ts):
                    break
                anchors.append(j)
                i = j
            if len(anchors) < 2:
                continue
            picked = np.array(anchors)
            v_pick = vals[picked]
            s0_list.append(v_pick[:-1])
            s1_list.append(v_pick[1:])

        if not s0_list:
            return None
        s0 = np.concatenate(s0_list)
        s1 = np.concatenate(s1_list)
        if len(s0) < MIN_PAIRS:
            return None
        A = np.column_stack([s0, np.ones_like(s0)])
        coef, *_ = np.linalg.lstsq(A, s1, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        a = min(max(a, 1e-6), 0.999999)
        if a >= 1.0 - 1e-9:
            return None
        mu = b / (1.0 - a)
        theta_per_ts = -math.log(a) / dt_target
        theta_per_day = theta_per_ts * TICKS_PER_DAY
        return {
            "dt_target": dt_target,
            "n_pairs":   len(s0),
            "a":         a,
            "mu":        mu,
            "theta_day": theta_per_day,
            "half_life_days": math.log(2) / theta_per_day,
        }

    print("# OU fit at multiple sampling intervals:")
    print(f"#   {'dt':>8}  {'n_pairs':>8}  {'a':>10}  {'mu':>10}  "
          f"{'theta/day':>10}  {'half-life (d)':>14}  {'conv':>6}")
    results = []
    for dt in sample_intervals:
        r = fit_one(dt)
        if r is None:
            print(f"#   {dt:>8}  -- insufficient data --")
            continue
        if results:
            t_prev = results[-1]["theta_day"]
            converged = abs(r["theta_day"] - t_prev) / max(r["theta_day"], 1e-12) < CONV_TOL
            r["converged"] = converged
        else:
            r["converged"] = False
        results.append(r)
        flag = "yes" if r["converged"] else "no"
        print(f"#   {r['dt_target']:>8}  {r['n_pairs']:>8}  "
              f"{r['a']:>10.6f}  {r['mu']:>10.4f}  "
              f"{r['theta_day']:>10.4f}  {r['half_life_days']:>14.4f}  {flag:>6}")

    if not results:
        raise ValueError("OU fit failed at every timescale")

    # Walk from longest backward; pick first one that converged
    chosen = None
    for r in reversed(results):
        if r["converged"]:
            chosen = r
            break

    if chosen is None:
        chosen = results[-1]
        print(f"# *** WARNING: theta not converged across timescales. "
              f"Falling back to longest available (dt={chosen['dt_target']}). ***")
        print(f"# *** This usually means more days of data are needed for a "
              f"reliable multi-day OU rate. ***")
    else:
        print(f"# Chosen timescale: dt={chosen['dt_target']} "
              f"(half-life {chosen['half_life_days']:.3f} days, converged)")

    return float(chosen["mu"]), float(chosen["theta_day"])


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


TARGET_TTE = 4.0   # TTE of the day you'll actually deploy on


def main():
    for p, _ in PATHS:
        if not Path(p).exists():
            sys.exit(f"missing file: {p}")

    loaded = [(load_day(p), day_idx, p) for p, day_idx in PATHS]

    # OU fit using per-row timestamp deltas; pass a list so transitions
    # don't span across days (cross-day deltas would be enormous).
    spot_series = [df["S"].dropna() for df, _, _ in loaded]
    mu, theta = ou_fit(spot_series)

    # Per-file WLS at a mid-day snapshot. Each (file, day_idx) gives one
    # (T, beta) sample. Multiple samples at the same T are kept — they
    # contribute to the within-T variance.
    samples, m_pool = [], []      # samples: list of (T, c0, c1, c2)
    for df, day_idx, path in loaded:
        full = df.dropna(how="any")
        if full.empty:
            continue
        ts_mid = full.index[len(full) // 2]
        snap = full.loc[ts_mid]
        S = float(snap["S"])
        T = max(TTE_START_DAYS - day_idx, 1e-9)
        res = wls_one_day(snap, S, T, mu, theta)
        if res is None:
            continue
        beta, m = res
        samples.append((T, beta[0], beta[1], beta[2]))
        m_pool.append(m)

    if len(samples) < 3:
        sys.exit("not enough valid daily fits for TTE extrapolation")

    arr = np.array(samples)        # cols: T, c0, c1, c2
    Ts  = arr[:, 0]
    A   = np.column_stack([Ts, np.ones_like(Ts)])

    # Linear regression of each coefficient against T. Extrapolation to
    # TARGET_TTE comes for free.
    slopes_intercepts = []
    x_target = []
    residuals_per_coef = []
    for col in range(1, 4):
        y = arr[:, col]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = float(sol[0]), float(sol[1])
        y_hat = slope * Ts + intercept
        slopes_intercepts.append((slope, intercept))
        x_target.append(slope * TARGET_TTE + intercept)
        residuals_per_coef.append(y - y_hat)

    x_target = np.array(x_target)

    # P0 = covariance of fit residuals (around the linear-in-T regression).
    # This captures variation NOT explained by the TTE trend — i.e. real
    # day-to-day smile noise the Kalman filter should track.
    R = np.column_stack(residuals_per_coef)        # (n_samples, 3)
    if len(samples) > 4:
        P0 = np.cov(R.T, ddof=1)
    else:
        P0 = np.cov(arr[:, 1:].T, ddof=1)          # fallback: raw covariance

    # Inflate P0 by extrapolation distance: variance grows ~linearly with
    # squared extrapolation distance from the training centroid.
    T_mean = float(Ts.mean())
    T_var  = float(((Ts - T_mean) ** 2).sum())
    extrap_factor = 1.0 + ((TARGET_TTE - T_mean) ** 2) / max(T_var, 1e-9)
    P0 = P0 * extrap_factor

    m_std = float(np.std(np.concatenate(m_pool)))

    # Also print the within-training average (X0_AVG) for sanity comparison
    x_avg = arr[:, 1:].mean(axis=0)

    # Emit constants
    print("# === Auto-generated by bootstrap.py ===")
    print(f"# {len(samples)} daily fits, TTE range [{Ts.min():.1f}, {Ts.max():.1f}]")
    print(f"# Coefficients extrapolated to TARGET_TTE = {TARGET_TTE}")
    print(f"# Extrapolation distance factor on P0: {extrap_factor:.3f}")
    print("import numpy as np")
    print()
    print("# Per-coefficient linear fits  c_i(T) = slope * T + intercept :")
    for i, (s, b) in enumerate(slopes_intercepts):
        print(f"#   c{i}(T) = {s:+.6e} * T + {b:+.6e}")
    print(f"# Training-set average (for reference, would be X0 if not extrapolating):")
    print(f"#   X0_AVG = [{x_avg[0]:+.6e}, {x_avg[1]:+.6e}, {x_avg[2]:+.6e}]")
    print()
    print(f"X0    = np.array([{x_target[0]:+.10e}, {x_target[1]:+.10e}, {x_target[2]:+.10e}])  "
          "# Level, Skew, Convexity @ TTE=" + f"{TARGET_TTE}")
    print("P0    = np.array([")
    for r in P0:
        print("    [" + ", ".join(f"{x:+.6e}" for x in r) + "],")
    print("])")
    print(f"M_STD          = {m_std:.6f}")
    print(f"OU_MU          = {mu:.4f}")
    print(f"OU_THETA       = {theta:.6e}  # per day")
    print(f"TTE_START_DAYS = {TARGET_TTE}")


if __name__ == "__main__":
    main()