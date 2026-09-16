"""
filter_pebbles.py
-----------------
Filters a semicolon-delimited CSV to only rows where product is a Purification Pebble.

Usage:
  python filter_pebbles.py prices.csv
  python filter_pebbles.py prices.csv --output pebbles.csv
"""

import argparse
import sys
import pandas as pd

PEBBLES = {"PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to input CSV")
    parser.add_argument("--output", default=None, help="Output file path (default: pebbles_<input>.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, sep=";")
    filtered = df[df["product"].isin(PEBBLES)]

    out = args.output or args.csv.replace(".csv", "_pebbles.csv")
    filtered.to_csv(out, sep=";", index=False)
    print(f"Rows in:  {len(df):,}")
    print(f"Rows out: {len(filtered):,}")
    print(f"Saved to: {out}")

if __name__ == "__main__":
    main()