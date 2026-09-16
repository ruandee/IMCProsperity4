"""
Grid search backtester for pebble.py
CSV format:
  day;timestamp;product;bid_price_1;bid_volume_1;...;ask_price_3;ask_volume_3;mid_price;profit_and_loss
"""

import csv
import math
import itertools
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ── paths ──────────────────────────────────────────────────────────────────────
CSV_FILES = [
    "imcprosperity4\dee/round5\prices_round_5_day_2.csv",
    "imcprosperity4\dee/round5\prices_round_5_day_3.csv",
    "imcprosperity4\dee/round5\prices_round_5_day_4.csv",
]

TRADE_PRODUCTS = ["PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
ALL_PRODUCTS   = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
POS_LIMIT      = 10

# ── grid ───────────────────────────────────────────────────────────────────────
GRID = {
    "WINDOW":   [20, 30, 44, 60],
    "MIN_OBS":  [10, 20],
    "ENTRY_Z":  [0.5, 1.0, 1.5, 2.0],
    "EXIT_Z":   [0.1, 0.2, 0.3],
    "STEP":     [1, 2, 3, 5],
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

class OrderDepth:
    """Minimal stand-in for the exchange datamodel."""
    def __init__(self):
        self.buy_orders:  Dict[int, int] = {}   # price → +volume
        self.sell_orders: Dict[int, int] = {}   # price → -volume


def load_ticks(csv_paths: List[str]) -> List[Dict[str, OrderDepth]]:
    """
    Returns a list of ticks, each tick being a dict  product → OrderDepth.
    Ticks are sorted globally by (day, timestamp).
    """
    rows = []
    for path in csv_paths:
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    rows.append(row)
        except FileNotFoundError:
            print(f"[WARN] {path} not found, skipping.")

    if not rows:
        raise FileNotFoundError("No CSV files loaded. Check CSV_FILES paths.")

    # Sort by day, then timestamp
    rows.sort(key=lambda r: (int(r["day"]), int(r["timestamp"])))

    # Group by (day, timestamp) → dict of product → OrderDepth
    ticks_dict: Dict[Tuple, Dict[str, OrderDepth]] = {}
    for row in rows:
        key = (int(row["day"]), int(row["timestamp"]))
        product = row["product"]

        if key not in ticks_dict:
            ticks_dict[key] = {}

        od = OrderDepth()
        for lvl in [1, 2, 3]:
            bp = row.get(f"bid_price_{lvl}", "").strip()
            bv = row.get(f"bid_volume_{lvl}", "").strip()
            ap = row.get(f"ask_price_{lvl}", "").strip()
            av = row.get(f"ask_volume_{lvl}", "").strip()

            if bp and bv:
                try: od.buy_orders[int(float(bp))] = int(float(bv))
                except ValueError: pass
            if ap and av:
                try:
                    # sell volumes are stored negative in the real datamodel
                    od.sell_orders[int(float(ap))] = -abs(int(float(av)))
                except ValueError: pass

        ticks_dict[key][product] = od

    return [ticks_dict[k] for k in sorted(ticks_dict)]


# ══════════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mean(x): return sum(x) / len(x)

def _std(x):
    if len(x) < 2: return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))

def _mid(od: OrderDepth) -> Optional[float]:
    b = max(od.buy_orders)  if od.buy_orders  else None
    a = min(od.sell_orders) if od.sell_orders else None
    # FIX: use explicit None checks, not truthiness (b=0 would be falsy)
    if b is not None and a is not None:
        return (b + a) / 2
    return None


def _fill(product: str, target: int, pos: int, od: OrderDepth
          ) -> Tuple[List[Tuple[int,int]], float]:
    """
    Returns (fills, pnl_impact).
    fills = list of (price, signed_qty) actually traded.
    """
    fills = []
    rem   = target - pos
    cost  = 0.0

    if rem > 0:                                          # need to buy
        for px, vol in sorted(od.sell_orders.items()):
            if rem <= 0: break
            avail = -vol                                 # vol is stored negative
            q = min(rem, avail)
            if q > 0:
                fills.append((px,  q))
                cost -= px * q
                rem  -= q
    elif rem < 0:                                        # need to sell
        for px, vol in sorted(od.buy_orders.items(), reverse=True):
            if rem >= 0: break
            q = min(-rem, vol)
            if q > 0:
                fills.append((px, -q))
                cost += px * q
                rem  += q

    return fills, cost


# ══════════════════════════════════════════════════════════════════════════════
# Single backtest run
# ══════════════════════════════════════════════════════════════════════════════

def backtest(ticks: List[Dict[str, OrderDepth]], params: dict) -> dict:
    WINDOW  = params["WINDOW"]
    MIN_OBS = params["MIN_OBS"]
    ENTRY_Z = params["ENTRY_Z"]
    EXIT_Z  = params["EXIT_Z"]
    STEP    = params["STEP"]

    spreads_hist: Dict[str, List[float]] = {p: [] for p in TRADE_PRODUCTS}
    pos:          Dict[str, int]         = {p: 0  for p in ALL_PRODUCTS}

    # Track which products we're actively holding (long / short)
    # FIX: explicit tracking so exit closes the *actual* held products
    held_long:  Optional[str] = None   # product we're long
    held_short: Optional[str] = None   # product we're short

    realized_pnl   = 0.0
    unrealized_pnl = 0.0
    trades         = 0
    pos_violations = 0

    for tick in ticks:
        mids = {}
        for p in ALL_PRODUCTS:
            od = tick.get(p)
            if od:
                m = _mid(od)
                if m is not None:
                    mids[p] = m

        if any(p not in mids for p in ALL_PRODUCTS):
            continue

        XS     = mids["PEBBLES_XS"]
        XL     = mids["PEBBLES_XL"]
        anchor = (50000 - (XS + XL)) / 3

        spreads_now = {}
        for p in TRADE_PRODUCTS:
            spread = mids[p] - anchor
            spreads_now[p] = spread
            spreads_hist[p].append(spread)
            spreads_hist[p] = spreads_hist[p][-WINDOW:]

        if any(len(spreads_hist[p]) < MIN_OBS for p in TRADE_PRODUCTS):
            continue

        zscores = {}
        skip = False
        for p in TRADE_PRODUCTS:
            s     = spreads_hist[p]
            mu    = _mean(s)
            sigma = _std(s)
            if sigma == 0:
                skip = True; break
            zscores[p] = (spreads_now[p] - mu) / sigma
        if skip:
            continue

        sorted_p = sorted(spreads_now, key=spreads_now.get)
        cheap = sorted_p[0]
        rich  = sorted_p[2]

        z_rich  = zscores[rich]
        z_cheap = zscores[cheap]

        def do_trade(product, target):
            nonlocal realized_pnl, trades, pos_violations
            od = tick.get(product)
            if od is None: return
            clamped = max(-POS_LIMIT, min(POS_LIMIT, target))
            if clamped != target:
                pos_violations += 1
            fills, cost = _fill(product, clamped, pos[product], od)
            if fills:
                pos[product] += sum(q for _, q in fills)
                realized_pnl += cost
                trades       += len(fills)

        # ── entry ──────────────────────────────────────────────────────────
        if z_rich > ENTRY_Z and z_cheap < -ENTRY_Z:
            target_rich  = max(-POS_LIMIT, pos[rich]  - STEP)
            target_cheap = min( POS_LIMIT, pos[cheap] + STEP)
            do_trade(rich,  target_rich)
            do_trade(cheap, target_cheap)
            held_short = rich
            held_long  = cheap

        # ── exit: close the products we actually hold ───────────────────────
        # FIX: use held_long/held_short instead of current rank
        elif (held_long is not None and held_short is not None
              and abs(zscores.get(held_long,  0)) < EXIT_Z
              and abs(zscores.get(held_short, 0)) < EXIT_Z):
            do_trade(held_long,  0)
            do_trade(held_short, 0)
            held_long  = None
            held_short = None

        # ── compute mark-to-market unrealized pnl ──────────────────────────
        unrealized_pnl = sum(pos[p] * mids[p] for p in ALL_PRODUCTS if p in mids)

    total_pnl = realized_pnl + unrealized_pnl
    return {
        "total_pnl":      round(total_pnl, 2),
        "realized_pnl":   round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "trades":         trades,
        "pos_violations": pos_violations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Grid search
# ══════════════════════════════════════════════════════════════════════════════

def grid_search(ticks, grid=GRID, top_n=10):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)
    print(f"Running {total} parameter combinations on {len(ticks)} ticks...\n")

    results = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))

        # Skip nonsensical combos
        if params["MIN_OBS"] >= params["WINDOW"]:
            continue
        if params["EXIT_Z"] >= params["ENTRY_Z"]:
            continue

        metrics = backtest(ticks, params)
        results.append({**params, **metrics})

        if i % max(1, total // 20) == 0:
            print(f"  {i}/{total} done, best so far: "
                  f"{max(r['total_pnl'] for r in results):.2f}")

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP {top_n} RESULTS")
    print(f"{'='*70}")

    header = (f"{'WIN':>4} {'OBS':>4} {'ENT':>5} {'EXT':>5} {'STP':>4} "
              f"{'TOTAL PNL':>12} {'REALIZED':>12} {'TRADES':>7} {'VIOLS':>6}")
    print(header)
    print("-" * len(header))

    for r in results[:top_n]:
        print(
            f"{r['WINDOW']:>4} {r['MIN_OBS']:>4} {r['ENTRY_Z']:>5.1f} "
            f"{r['EXIT_Z']:>5.2f} {r['STEP']:>4} "
            f"{r['total_pnl']:>12.2f} {r['realized_pnl']:>12.2f} "
            f"{r['trades']:>7} {r['pos_violations']:>6}"
        )

    print(f"\nBest params: { {k: results[0][k] for k in keys} }")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ticks = load_ticks(CSV_FILES)
    print(f"Loaded {len(ticks)} ticks.\n")
    all_results = grid_search(ticks, top_n=15)

    # Save full results to CSV
    import csv as csv_mod
    out = "grid_results.csv"
    fieldnames = list(GRID.keys()) + ["total_pnl", "realized_pnl",
                                       "unrealized_pnl", "trades", "pos_violations"]
    with open(out, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nFull results saved to {out}")