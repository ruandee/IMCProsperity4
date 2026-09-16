#!/usr/bin/env python3
"""
Convert a Prosperity-style submission .log file into CSVs.

The .log file is a single JSON object with these keys:
  - submissionId   : str
  - activitiesLog  : str  (semicolon-delimited table of order-book snapshots)
  - logs           : list of {timestamp, sandboxLog, lambdaLog}
  - tradeHistory   : list of {timestamp, buyer, seller, symbol, currency, price, quantity}

Outputs three files alongside the input (or in --out-dir):
  <stem>_activities.csv
  <stem>_logs.csv
  <stem>_trades.csv

Usage:
  python log_to_csv.py path/to/file.log
  python log_to_csv.py path/to/file.log --out-dir ./csvs
  python log_to_csv.py *.log --out-dir ./csvs
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def write_activities_csv(activities_log: str, out_path: Path) -> int:
    """activitiesLog is already a semicolon-delimited table with a header row.
    Re-emit it as a proper comma-delimited CSV (handles quoting if needed)."""
    rows_written = 0
    lines = activities_log.splitlines()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for line in lines:
            if not line.strip():
                continue
            writer.writerow(line.split(";"))
            rows_written += 1
    return rows_written  # includes header


def write_logs_csv(logs: list, out_path: Path, drop_empty: bool = True) -> int:
    """Each log entry has timestamp, sandboxLog, lambdaLog. Most are blank;
    by default we drop entries where both log fields are empty."""
    fieldnames = ["timestamp", "sandboxLog", "lambdaLog"]
    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in logs:
            sandbox = entry.get("sandboxLog", "")
            lambda_ = entry.get("lambdaLog", "")
            if drop_empty and not sandbox and not lambda_:
                continue
            writer.writerow({
                "timestamp": entry.get("timestamp", ""),
                "sandboxLog": sandbox,
                "lambdaLog": lambda_,
            })
            rows_written += 1
    return rows_written


def write_trades_csv(trade_history: list, out_path: Path) -> int:
    fieldnames = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]
    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trade_history:
            writer.writerow({k: trade.get(k, "") for k in fieldnames})
            rows_written += 1
    return rows_written


def convert_one(log_path: Path, out_dir: Path, keep_empty_logs: bool) -> None:
    print(f"Reading {log_path}")
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stem = log_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    activities_path = out_dir / f"{stem}_activities.csv"
    logs_path = out_dir / f"{stem}_logs.csv"
    trades_path = out_dir / f"{stem}_trades.csv"

    if "activitiesLog" in data and data["activitiesLog"]:
        n = write_activities_csv(data["activitiesLog"], activities_path)
        print(f"  wrote {activities_path.name}  ({n - 1} data rows)")
    else:
        print("  no activitiesLog found, skipping")

    if "logs" in data:
        n = write_logs_csv(data["logs"], logs_path, drop_empty=not keep_empty_logs)
        print(f"  wrote {logs_path.name}        ({n} rows)")

    if "tradeHistory" in data:
        n = write_trades_csv(data["tradeHistory"], trades_path)
        print(f"  wrote {trades_path.name}      ({n} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_files", nargs="+", type=Path, help="One or more .log files")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same dir as each input file)")
    parser.add_argument("--keep-empty-logs", action="store_true",
                        help="Keep log entries where both sandboxLog and lambdaLog are blank "
                             "(default: drop them, which usually shrinks the logs CSV by ~99%%)")
    args = parser.parse_args()

    for log_path in args.log_files:
        if not log_path.is_file():
            print(f"Skipping {log_path}: not a file", file=sys.stderr)
            continue
        out_dir = args.out_dir if args.out_dir is not None else log_path.parent
        try:
            convert_one(log_path, out_dir, args.keep_empty_logs)
        except json.JSONDecodeError as e:
            print(f"Error parsing {log_path}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())