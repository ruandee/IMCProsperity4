"""
OU mean-reversion / sigma fit for VEV_X options per day, plotted as
volatility vs log-moneyness ln(K / F_underlying).

CSV schema (semicolon-separated):
  day;timestamp;product;bid_price_1;bid_volume_1;...;mid_price;profit_and_loss

Strikes:    VEV_4000, VEV_4500, VEV_5100..5500, VEV_6000, VEV_6500
Underlying: VELVETFRUIT_EXTRACT  (used only to compute moneyness; not fit)
Days 0..3 -> TTE 8,7,6,5  (one trading-day = 1 unit of time in TTE)

Usage:
    python ou_vol_smile.py path/to/data.csv [--out outdir]
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- config ----------------------------------------------------------------
STRIKES = [4000, 4500, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
UNDERLYING = "VELVETFRUIT_EXTRACT"
PRODUCTS = [f"VEV_{k}" for k in STRIKES]
DAY_TO_TTE = {0: 8, 1: 7, 2: 6, 3: 5}


# --- OU MLE ----------------------------------------------------------------
def fit_ou(x, dt=1.0):
    """
    Exact-discretization MLE for dX = theta*(mu - X)dt + sigma dW.

    Uses AR(1):  x_{t+1} = a + b*x_t + eps,  eps ~ N(0, s_eps^2)
        b     = e^{-theta*dt}
        a     = mu * (1 - b)
        s_eps^2 = sigma^2 * (1 - b^2) / (2*theta)

    Inversion:
        theta = -ln(b) / dt
        mu    = a / (1 - b)
        sigma = s_eps * sqrt(-2 ln(b) / (dt * (1 - b^2)))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        return np.nan, np.nan, np.nan, np.nan

    x0, x1 = x[:-1], x[1:]
    n = x1.size

    sx, sy = x0.sum(), x1.sum()
    sxx = (x0 * x0).sum()
    sxy = (x0 * x1).sum()
    syy = (x1 * x1).sum()

    denom = n * sxx - sx * sx
    if denom <= 0:
        return np.nan, np.nan, np.nan, np.nan

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    # residual variance (unbiased)
    resid_ss = syy - a * sy - b * sxy
    s_eps2 = max(resid_ss / max(n - 2, 1), 1e-18)

    # Need 0 < b < 1 for a stationary, mean-reverting fit.
    if not (0.0 < b < 1.0):
        # Fall back to half-life from autocorr clamped; report sigma from
        # plain return std as a robust proxy.
        sigma_proxy = np.sqrt(s_eps2 / dt)
        return np.nan, np.nan, sigma_proxy, np.nan

    theta = -np.log(b) / dt
    mu = a / (1.0 - b)
    sigma = np.sqrt(s_eps2 * (-2.0 * np.log(b)) / (dt * (1.0 - b * b)))
    half_life = np.log(2.0) / theta
    return theta, mu, sigma, half_life


# --- data loading ----------------------------------------------------------
def load(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    needed = {"day", "timestamp", "product", "mid_price"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["day"] = df["day"].astype(int)
    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    df = df.dropna(subset=["mid_price"])
    df = df.sort_values(["day", "timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def strike_from_product(p):
    m = re.match(r"VEV_(\d+)$", str(p))
    return int(m.group(1)) if m else None


# --- per-day fit -----------------------------------------------------------
def fit_day(df_day):
    """
    Returns DataFrame with one row per strike: theta, mu, sigma, half_life,
    log-moneyness, n_obs.
    """
    # forward F: use mid of underlying on this day (mean of mids; OU mean
    # would also be reasonable, but mean is robust and bias-free for moneyness)
    und = df_day[df_day["product"] == UNDERLYING]["mid_price"]
    F = float(und.mean()) if len(und) else np.nan

    rows = []
    for prod in PRODUCTS:
        sub = df_day[df_day["product"] == prod]["mid_price"].to_numpy()
        K = strike_from_product(prod)
        theta, mu, sigma, hl = fit_ou(sub, dt=1.0)
        log_m = np.log(K / F) if (F and F > 0) else np.nan
        rows.append(
            dict(product=prod, strike=K, n=len(sub), F=F, log_moneyness=log_m,
                 theta=theta, mu=mu, sigma=sigma, half_life=hl)
        )
    return pd.DataFrame(rows)


# --- rolling IV fit -------------------------------------------------------
def fit_ou_rolling(prices, window=500, dt=1.0):
    """
    Fit OU over rolling window, return array of sigmas aligned with prices.
    If window > prices, returns array of NaNs.
    """
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < window + 1:
        return np.full(len(prices), np.nan)
    
    sigmas = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        sub = prices[i - window : i + 1]
        _, _, sigma, _ = fit_ou(sub, dt=dt)
        sigmas[i] = sigma
    return sigmas


# --- plotting: IV timeseries per strike per day ---------------------------
def plot_iv_timeseries(df_day, day, tte, out_path, window=50):
    """
    For each strike, plot rolling-window OU sigma vs time.
    Creates a grid (3 cols) of subplots.
    """
    strikes_in_day = [s for s in STRIKES if any(df_day["product"] == f"VEV_{s}")]
    n_strikes = len(strikes_in_day)
    if n_strikes == 0:
        return
    
    ncols = 3
    nrows = (n_strikes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    if n_strikes == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, K in enumerate(strikes_in_day):
        sub = df_day[df_day["product"] == f"VEV_{K}"]["mid_price"].to_numpy()
        sigmas = fit_ou_rolling(sub, window=window, dt=1.0)
        ax = axes[idx]
        ax.plot(range(len(sigmas)), sigmas, linewidth=1.5, alpha=0.8, color='steelblue')
        ax.fill_between(range(len(sigmas)), sigmas, alpha=0.2, color='steelblue')
        ax.set_title(f"Strike {K}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (obs)")
        ax.set_ylabel("OU σ (rolling window)")
        ax.grid(True, alpha=0.3)
    
    # hide unused subplots
    for idx in range(n_strikes, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Day {day} (TTE={tte}) – IV over Time  |  Window={window}", 
                 fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --- plotting: half-life across days per strike --------------------------
def plot_half_life_across_days(summary_rows, out_path):
    """
    Plot half-life vs day for each strike.
    One line per strike.
    """
    full = pd.concat(summary_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for K in STRIKES:
        sub = full[full["strike"] == K].sort_values("day")
        sub = sub.dropna(subset=["half_life"])
        if len(sub) > 0:
            ax.plot(sub["day"], sub["half_life"], marker="o", label=f"K={K}", linewidth=1.8)
    
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Half-Life (timesteps)", fontsize=11)
    ax.set_title("Mean Reversion Half-Life per Strike across Days", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --- plotting: theta across days per strike ------------------------------
def plot_theta_across_days(summary_rows, out_path):
    """
    Plot mean-reversion speed (theta) vs day for each strike.
    One line per strike.
    """
    full = pd.concat(summary_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for K in STRIKES:
        sub = full[full["strike"] == K].sort_values("day")
        sub = sub.dropna(subset=["theta"])
        if len(sub) > 0:
            ax.plot(sub["day"], sub["theta"], marker="s", label=f"K={K}", linewidth=1.8)
    
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("θ (mean reversion speed)", fontsize=11)
    ax.set_title("Mean Reversion Speed per Strike across Days", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --- plotting: all IV timeseries together (all days, all strikes) -----------
def plot_all_iv_combined(df, out_path, window=50):
    """
    Plot rolling OU sigma vs time for all strikes across all days.
    Each strike gets a unique color; lines are concatenated across days.
    """
    import matplotlib.colors as mcolors
    
    # Get a color map with enough colors for all strikes
    try:
        cmap = plt.colormaps.get_cmap('tab20')
    except:
        cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(idx / len(STRIKES)) for idx in range(len(STRIKES))]
    color_map = {strike: colors[idx] for idx, strike in enumerate(STRIKES)}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    time_offset = 0  # track position along x-axis
    day_boundaries = []
    
    for day in sorted(df["day"].unique()):
        df_day = df[df["day"] == day]
        n_obs_in_day = len(df_day[df_day["product"] == UNDERLYING])
        
        for strike in STRIKES:
            sub = df_day[df_day["product"] == f"VEV_{strike}"]["mid_price"].to_numpy()
            if len(sub) > 0:
                sigmas = fit_ou_rolling(sub, window=window, dt=1.0)
                x_vals = np.arange(time_offset, time_offset + len(sigmas))
                ax.plot(x_vals, sigmas, linewidth=1.5, alpha=0.75, 
                       color=color_map[strike], label=f"K={strike}" if day == 0 else "")
        
        day_boundaries.append(time_offset)
        time_offset += n_obs_in_day
    
    # Add vertical lines at day boundaries
    for boundary in day_boundaries[1:]:
        ax.axvline(boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Custom legend: one entry per strike
    handles = [plt.Line2D([0], [0], color=color_map[k], linewidth=2) for k in STRIKES]
    ax.legend(handles, [f"K={k}" for k in STRIKES], 
             loc='best', fontsize=9, ncol=3, framealpha=0.95)
    
    ax.set_xlabel("Time (observations across all days)", fontsize=12)
    ax.set_ylabel("OU σ (rolling window)", fontsize=12)
    ax.set_title("IV (OU σ) for All Strikes over All Days  |  Window=50", 
                fontsize=13, fontweight='bold')
    ax.set_ylim(top=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to CSV")
    ap.add_argument("--out", default="ou_out", help="Output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)

    summary_rows = []
    for day in sorted(DAY_TO_TTE):
        sub = df[df["day"] == day]
        if sub.empty:
            print(f"[skip] no rows for day {day}")
            continue
        res = fit_day(sub)
        res.insert(0, "day", day)
        res.insert(1, "TTE", DAY_TO_TTE[day])
        res.to_csv(out / f"ou_fit_day{day}.csv", index=False)
        # Plot IV over time per strike
        plot_iv_timeseries(sub, day, DAY_TO_TTE[day], out / f"iv_timeseries_day{day}.png", window=50)
        summary_rows.append(res)
        print(f"[ok] day {day}: fit {res['sigma'].notna().sum()}/{len(res)} strikes")

    if summary_rows:
        full = pd.concat(summary_rows, ignore_index=True)
        full.to_csv(out / "ou_fit_all.csv", index=False)
        
        # Plot half-life and theta across days
        plot_half_life_across_days(summary_rows, out / "half_life_across_days.png")
        plot_theta_across_days(summary_rows, out / "theta_across_days.png")
        
        # Plot all IV together
        plot_all_iv_combined(df, out / "all_iv_combined.png", window=50)
        
        print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()