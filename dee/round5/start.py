"""
plot_baskets.py
---------------
Reads a semicolon-delimited CSV with columns:
  day;timestamp;product;bid_price_1;bid_volume_1;...;mid_price;profit_and_loss

Produces:
  • One figure per basket (10 total) — 5 product mid-price lines + basket average line
  • One summary figure — all 10 basket average lines together

Usage:
  python plot_baskets.py prices.csv
  python plot_baskets.py prices.csv --save          # saves PNGs instead of showing
  python plot_baskets.py prices.csv --output ./out  # output directory for PNGs
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display, no MemoryError
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Palette ───────────────────────────────────────────────────────────────────
PRODUCT_COLORS = ["#4E9AF1", "#F4A93D", "#6BCB77", "#FF6B6B", "#C77DFF"]
BASKET_COLOR   = "#FFFFFF"        # white basket-average line on dark bg
DAY_ALPHA      = [1.0, 0.65, 0.35]  # days 2, 3, 4 – lighter for older days

DARK_BG   = "#0D1117"
PANEL_BG  = "#161B22"
GRID_CLR  = "#30363D"
TEXT_CLR  = "#E6EDF3"
SPINE_CLR = "#30363D"

SUMMARY_COLORS = [
    "#4E9AF1","#F4A93D","#6BCB77","#FF6B6B","#C77DFF",
    "#FF9F45","#00C4CC","#F72585","#7FFF00","#FFD700",
]

# ── Basket definitions ─────────────────────────────────────────────────────────
BASKETS = {
    "Galaxy Sounds Recorders": [
        "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "Vertical Sleeping Pods": [
        "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    ],
    "Organic Microchips": [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    ],
    "Purification Pebbles": [
        "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    ],
    "Domestic Robots": [
        "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
        "ROBOT_LAUNDRY", "ROBOT_IRONING",
    ],
    "UV-Visors": [
        "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
        "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    ],
    "Instant Translators": [
        "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "Construction Panels": [
        "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    ],
    "Liquid Breath Oxygen Shakes": [
        "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    ],
    "Protein Snack Packs": [
        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
    ],
}

DAYS = [2, 3, 4]


# ── Helpers ────────────────────────────────────────────────────────────────────
def apply_dark_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.4, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def day_x_offset(day_val):
    """Convert day number to a starting x-offset so days appear concatenated."""
    # Each day's timestamps are assumed to start near 0 → offset by day index * max_ts
    # We'll handle this dynamically in the main loop.
    return 0


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()
    # Keep only days 2, 3, 4
    df = df[df["day"].isin(DAYS)].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    return df


def build_continuous_x(df: pd.DataFrame) -> pd.DataFrame:
    """Create a monotonically increasing x-axis across days by appending per-day
    ts ranges one after the other."""
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    offset = 0
    chunks = []
    for day in sorted(df["day"].unique()):
        chunk = df[df["day"] == day].copy()
        chunk["x"] = chunk["timestamp"] + offset
        offset = chunk["x"].max() + 1
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def day_boundaries(df: pd.DataFrame):
    """Return list of (day_label, x_start, x_end) for shading."""
    bounds = []
    for day in sorted(df["day"].unique()):
        sub = df[df["day"] == day]
        bounds.append((day, sub["x"].min(), sub["x"].max()))
    return bounds


def shade_days(ax, bounds, ymin, ymax):
    day_shades = ["#1C2333", "#131A27", "#0D1117"]
    for i, (day, xlo, xhi) in enumerate(bounds):
        ax.axvspan(xlo, xhi, alpha=0.25, color=day_shades[i % 3], zorder=0)
        ax.text(
            (xlo + xhi) / 2, ymax, f"Day {day}",
            ha="center", va="top", fontsize=6, color="#8B949E",
            transform=ax.get_xaxis_transform()
        )


# ── Per-basket figure ──────────────────────────────────────────────────────────
def plot_basket(basket_name: str, products: list, df_full: pd.DataFrame,
                save_dir: str):
    """One figure: 5-product lines + basket average, coloured by day."""

    # Filter to products in this basket
    df = df_full[df_full["product"].isin(products)].copy()
    df = df.dropna(subset=["mid_price"])
    if df.empty or len(df) < 2:
        print(f"  [WARN] Insufficient data for basket '{basket_name}'. Skipping.")
        return

    df = build_continuous_x(df)
    bounds = day_boundaries(df)

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor(DARK_BG)
    apply_dark_style(ax)

    ymin, ymax = np.inf, -np.inf

    # Plot each product
    for i, product in enumerate(products):
        pdata = df[df["product"] == product].sort_values("x")
        if pdata.empty:
            continue
        ax.plot(
            pdata["x"], pdata["mid_price"],
            color=PRODUCT_COLORS[i], linewidth=1.0, alpha=0.85,
            label=product.replace("_", " ").title(),
            zorder=3,
        )
        ymin = min(ymin, pdata["mid_price"].min())
        ymax = max(ymax, pdata["mid_price"].max())

    # Basket average — pivot then mean across products
    pivot = df.pivot_table(index="x", columns="product", values="mid_price", aggfunc="mean")
    basket_avg = pivot.mean(axis=1)
    ax.plot(
        basket_avg.index, basket_avg.values,
        color=BASKET_COLOR, linewidth=2.2, alpha=0.95,
        linestyle="--", label="Basket Average",
        zorder=5,
    )
    ymin = min(ymin, basket_avg.min())
    ymax = max(ymax, basket_avg.max())

    # Day shading
    pad = (ymax - ymin) * 0.05 if ymax != ymin else 1
    shade_days(ax, bounds, ymin - pad, ymax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(basket_name, fontsize=13, fontweight="bold", pad=10, color=TEXT_CLR)
    ax.set_xlabel("Timestamp (continuous across days)", fontsize=8)
    ax.set_ylabel("Mid Price", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    legend = ax.legend(
        loc="upper left", fontsize=7, framealpha=0.3,
        facecolor=PANEL_BG, edgecolor=SPINE_CLR, labelcolor=TEXT_CLR,
        ncol=2,
    )

    # layout handled by constrained_layout=True

    safe_name = basket_name.replace(" ", "_").lower()
    out = Path(save_dir) / f"basket_{safe_name}.png"
    print(f"  Saving …")
    fig.savefig(out, dpi=96, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Summary figure (all 10 baskets) ───────────────────────────────────────────
def plot_all_baskets(all_basket_avgs: dict, save_dir: str):
    """One figure with all 10 basket average mid-price lines."""

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    fig.patch.set_facecolor(DARK_BG)
    apply_dark_style(ax)

    # Determine global day bounds from the first basket that has data
    any_df = next(iter(all_basket_avgs.values()))
    bounds = [(day, any_df[any_df["day"] == day]["x"].min(),
               any_df[any_df["day"] == day]["x"].max())
              for day in sorted(any_df["day"].unique())]

    ymin, ymax = np.inf, -np.inf

    for i, (bname, df_avg) in enumerate(all_basket_avgs.items()):
        pivot = df_avg.pivot_table(index="x", columns="product", values="mid_price", aggfunc="mean")
        basket_avg = pivot.mean(axis=1)
        ax.plot(
            basket_avg.index, basket_avg.values,
            color=SUMMARY_COLORS[i % len(SUMMARY_COLORS)],
            linewidth=1.6, alpha=0.9,
            label=bname,
            zorder=3,
        )
        ymin = min(ymin, basket_avg.min())
        ymax = max(ymax, basket_avg.max())

    pad = (ymax - ymin) * 0.05 if ymax != ymin else 1
    shade_days(ax, bounds, ymin - pad, ymax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title("All 10 Basket Average Prices", fontsize=14, fontweight="bold",
                 pad=10, color=TEXT_CLR)
    ax.set_xlabel("Timestamp (continuous across days)", fontsize=9)
    ax.set_ylabel("Basket Average Mid Price", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax.legend(
        loc="upper left", fontsize=8, framealpha=0.35,
        facecolor=PANEL_BG, edgecolor=SPINE_CLR, labelcolor=TEXT_CLR,
        ncol=2,
    )

    # layout handled by constrained_layout=True

    out = Path(save_dir) / "all_baskets_summary.png"
    fig.savefig(out, dpi=96, facecolor=DARK_BG)
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot basket mid-price time series.")
    parser.add_argument("csv", help="Path to the semicolon-delimited CSV file")
    parser.add_argument("--output", default="charts",
                        help="Directory to save PNGs (default: ./charts)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"Error: file not found — {args.csv}")

    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(save_dir)}")

    print(f"\nLoading {args.csv} …")
    df_raw = load_data(args.csv)
    print(f"  Rows loaded: {len(df_raw):,}")
    print(f"  Days present: {sorted(df_raw['day'].unique())}")
    print(f"  Products found: {df_raw['product'].nunique()}")

    all_basket_avgs = {}

    for i, (basket_name, products) in enumerate(BASKETS.items(), 1):
        print(f"\n[{i:02d}/10] {basket_name}")
        df_basket = df_raw[df_raw["product"].isin(products)].copy()
        if df_basket.empty:
            print("  → no data, skipping")
            continue
        df_basket = build_continuous_x(df_basket)
        all_basket_avgs[basket_name] = df_basket
        plot_basket(basket_name, products, df_raw, save_dir=save_dir)

    print("\n[11/11] Summary — all 10 baskets")
    plot_all_baskets(all_basket_avgs, save_dir=save_dir)

    print(f"\nDone. 11 PNGs written to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()