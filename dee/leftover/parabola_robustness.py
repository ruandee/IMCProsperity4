#!/usr/bin/env python3
"""
Compute implied volatility for each VEV_X strike at every timestamp,
then plot the IV smile (IV vs moneyness) with the fitted parabola overlay.

Assumptions
-----------
- VEV_X are European *calls* on VELVETFRUIT_EXTRACT (the integer suffix is K).
- TTE = 5 (in whatever time unit the parabola was fitted in).
- Risk-free rate r = 0.
- Mid price (mid_price column) is used as the option price and as S.
- Moneyness M = K / S.

Usage
-----
    python iv_smile.py filtered_activities.csv
    python iv_smile.py filtered_activities.csv --tte 5 --out-prefix run1

Outputs
-------
    <prefix>_iv.csv     # one row per (timestamp, strike) with M and IV
    <prefix>_smile.png  # the plot
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- config -----------------------------------------------------------------
UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKE_RE = re.compile(r"^VEV_(\d+)$")
PARABOLA_COEFFS = (0.051280, -0.103242, 0.064080)  # a, b, c  ->  a*M^2 + b*M + c


# --- Black-Scholes ----------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    """Vectorised Black-Scholes call price."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * norm.pdf(d1) * sqrtT


def implied_vol(price, S, K, T, r=0.0, max_iter=80, tol=1e-8):
    """
    Vectorised Newton-Raphson IV solver for European calls.
    Returns NaN for rows where the price violates no-arbitrage bounds or
    the iteration fails to converge to a sensible value.
    """
    price = np.asarray(price, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # No-arb bounds for a European call: max(S - K*e^{-rT}, 0) <= C <= S
    intrinsic = np.maximum(S - K * np.exp(-r * T), 0.0)
    valid = (price > intrinsic + 1e-9) & (price < S - 1e-9)

    sigma = np.full_like(price, 0.2)  # initial guess
    for _ in range(max_iter):
        diff = bs_call(S, K, T, r, sigma) - price
        v = bs_vega(S, K, T, r, sigma)
        v = np.where(v < 1e-12, 1e-12, v)  # guard
        step = diff / v
        sigma_new = np.clip(sigma - step, 1e-6, 5.0)
        if np.max(np.abs(sigma_new - sigma)) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new

    # final residual check — drop anything that didn't converge
    resid = bs_call(S, K, T, r, sigma) - price
    converged = np.abs(resid) < 1e-3
    sigma = np.where(valid & converged, sigma, np.nan)
    return sigma


# --- main pipeline ----------------------------------------------------------
def extract_strike(product: str):
    m = STRIKE_RE.match(str(product))
    return int(m.group(1)) if m else np.nan


def build_iv_table(df: pd.DataFrame, tte: float) -> pd.DataFrame:
    # underlying mid per timestamp
    under = (
        df.loc[df["product"] == UNDERLYING, ["timestamp", "mid_price"]]
        .rename(columns={"mid_price": "S"})
        .drop_duplicates("timestamp")
    )

    # option rows
    opts = df[df["product"].str.match(r"^VEV_\d+$", na=False)].copy()
    opts["K"] = opts["product"].map(extract_strike)
    opts = opts.dropna(subset=["mid_price", "K"])

    merged = opts.merge(under, on="timestamp", how="inner")
    merged = merged.dropna(subset=["S"])

    merged["IV"] = implied_vol(
        merged["mid_price"].to_numpy(),
        merged["S"].to_numpy(),
        merged["K"].to_numpy(),
        T=tte,
    )
    merged["M"] = merged["K"] / merged["S"]

    return merged[["timestamp", "product", "K", "S", "mid_price", "M", "IV"]]


def plot_smile(iv_df: pd.DataFrame, out_path: Path) -> None:
    iv_df = iv_df.dropna(subset=["IV"])
    strikes = sorted(iv_df["K"].unique())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("viridis")

    # one scatter per strike (each strike traces a tight cluster in M)
    for i, K in enumerate(strikes):
        sub = iv_df[iv_df["K"] == K].sort_values("M")
        color = cmap(i / max(len(strikes) - 1, 1))
        ax.scatter(
            sub["M"], sub["IV"],
            s=4, alpha=0.35, color=color, label=f"VEV_{int(K)}",
        )

    # parabola overlay
    a, b, c = PARABOLA_COEFFS
    m_min, m_max = iv_df["M"].min(), iv_df["M"].max()
    span = m_max - m_min
    M_grid = np.linspace(m_min - 0.05 * span, m_max + 0.05 * span, 400)
    parab = a * M_grid**2 + b * M_grid + c
    ax.plot(M_grid, parab, color="red", linewidth=2.2,
            label=f"{a:.6f}·M² + {b:.6f}·M + {c:.6f}")

    ax.set_xlabel("Moneyness  M = K / S")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Implied volatility smile — VEV_X options")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="filtered_activities.csv")
    parser.add_argument("--tte", type=float, default=5.0, help="time to expiry (default 5)")
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="output file prefix (default: input stem)")
    args = parser.parse_args()

    prefix = args.out_prefix or args.csv_path.with_suffix("")
    iv_csv = Path(f"{prefix}_iv.csv")
    plot_png = Path(f"{prefix}_smile.png")

    print(f"reading {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    print("computing IVs...")
    iv_df = build_iv_table(df, tte=args.tte)
    n_total = len(iv_df)
    n_valid = iv_df["IV"].notna().sum()
    print(f"  {n_valid}/{n_total} rows produced a valid IV")

    iv_df.to_csv(iv_csv, index=False)
    print(f"wrote {iv_csv}")

    plot_smile(iv_df, plot_png)
    print(f"wrote {plot_png}")


if __name__ == "__main__":
    main()