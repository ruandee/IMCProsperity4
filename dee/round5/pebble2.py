from datamodel import OrderDepth, TradingState, Order
from typing import Dict, Optional
import json
import math


TRADE_PRODUCTS = ["PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
ALL_PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]

POS_LIMIT = 10
WINDOW = 45
MIN_OBS = 15
ENTRY_Z = 1.5
EXIT_Z = 0.4
K = 2

# Require real edge against our tracked entry before crossing an exit.
EXIT_EDGE_TICKS = 1

# Bollinger threshold adapts to recent spread volatility.
VOL_WINDOW = 60
MIN_ENTRY_Z = 0.6
MAX_ENTRY_Z = 3.0

# Size ramps harder once the signal moves from strong to extreme.
RAMP_Z_START = 1.55
RAMP_Z_FULL = 2.8
MAX_SIZE_MULT = 1.5

# MM now runs on all three products simultaneously.
# Limits and qty are scaled down accordingly; skew is tightened to
# manage inventory faster across three concurrent books.
MM_POS_LIMIT = 10
MM_QTY = 1
MM_EDGE = 2
MM_INV_SKEW = 3


def _mean(x):
    return sum(x) / len(x)


def _std(x):
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


class Trader:

    @staticmethod
    def _mid(od: Optional[OrderDepth]) -> Optional[float]:
        if od is None:
            return None
        if not od.buy_orders or not od.sell_orders:
            return None

        bid_vol = sum(vol for vol in od.buy_orders.values() if vol > 0)
        ask_vol = sum(-vol for vol in od.sell_orders.values() if vol < 0)
        if bid_vol <= 0 or ask_vol <= 0:
            return None

        bid_vwap = sum(px * vol for px, vol in od.buy_orders.items() if vol > 0) / bid_vol
        ask_vwap = sum(px * -vol for px, vol in od.sell_orders.items() if vol < 0) / ask_vol
        return (bid_vwap + ask_vwap) / 2

    @staticmethod
    def _cross(product, target, pos, od, orders):
        """Aggressive: lift offers / hit bids to guarantee fill."""
        rem = target - pos
        if rem > 0:
            for px, vol in sorted(od.sell_orders.items()):
                if rem <= 0:
                    break
                q = min(rem, -vol)
                if q > 0:
                    orders.append(Order(product, px, q))
                    rem -= q
        elif rem < 0:
            for px, vol in sorted(od.buy_orders.items(), reverse=True):
                if rem >= 0:
                    break
                q = min(-rem, vol)
                if q > 0:
                    orders.append(Order(product, px, -q))
                    rem += q

    @staticmethod
    def _post(product, target, pos, od, orders):
        """Passive: rest inside the spread, do not cross."""
        rem = target - pos
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        if rem > 0 and best_ask is not None:
            px = best_ask - 1
            if best_bid is None or px > best_bid:
                orders.append(Order(product, px, rem))

        elif rem < 0 and best_bid is not None:
            px = best_bid + 1
            if best_ask is None or px < best_ask:
                orders.append(Order(product, px, rem))

    @staticmethod
    def _signed_trade_qty(trade) -> int:
        qty = int(trade.quantity)
        if getattr(trade, "buyer", None) == "SUBMISSION":
            return abs(qty)
        if getattr(trade, "seller", None) == "SUBMISSION":
            return -abs(qty)
        return qty

    @staticmethod
    def _apply_fill_to_avg(pos: int, avg_px: float, price: float, qty: int):
        if qty == 0:
            return pos, avg_px

        same_direction = (
            pos == 0 or
            (pos > 0 and qty > 0) or
            (pos < 0 and qty < 0)
        )

        if same_direction:
            new_pos = pos + qty
            if new_pos == 0:
                return 0, 0.0
            new_avg = ((abs(pos) * avg_px) + (abs(qty) * price)) / abs(new_pos)
            return new_pos, new_avg

        new_pos = pos + qty
        if new_pos == 0:
            return 0, 0.0
        if (pos > 0 and new_pos > 0) or (pos < 0 and new_pos < 0):
            return new_pos, avg_px

        # Position flipped; leftover inventory starts fresh at the fill price.
        return new_pos, float(price)

    def _reconcile_entry_state(self, data, state, mids):
        entry_avg = data.get("entry_avg", {p: 0.0 for p in TRADE_PRODUCTS})
        tracked_pos = data.get("tracked_pos", {p: 0 for p in TRADE_PRODUCTS})

        for p in TRADE_PRODUCTS:
            avg_px = float(entry_avg.get(p, 0.0))
            tracked = int(tracked_pos.get(p, 0))

            for trade in state.own_trades.get(p, []):
                qty = self._signed_trade_qty(trade)
                tracked, avg_px = self._apply_fill_to_avg(
                    tracked, avg_px, trade.price, qty
                )

            actual = int(state.position.get(p, 0))
            if actual == 0:
                tracked, avg_px = 0, 0.0
            elif tracked != actual:
                # If traderData is out of sync, keep exits conservative.
                tracked = actual
                if avg_px <= 0:
                    avg_px = mids.get(p, 0.0)

            tracked_pos[p] = tracked
            entry_avg[p] = avg_px

        data["entry_avg"] = entry_avg
        data["tracked_pos"] = tracked_pos
        return entry_avg

    @staticmethod
    def _cross_exit_profitable(pos: int, avg_px: float, od: OrderDepth) -> bool:
        if avg_px <= 0:
            return False

        if pos > 0:
            best_bid = max(od.buy_orders) if od.buy_orders else None
            return best_bid is not None and best_bid >= avg_px + EXIT_EDGE_TICKS

        if pos < 0:
            best_ask = min(od.sell_orders) if od.sell_orders else None
            return best_ask is not None and best_ask <= avg_px - EXIT_EDGE_TICKS

        return False

    @staticmethod
    def _post_profitable_exit(product, target, pos, avg_px, od, orders):
        if avg_px <= 0:
            return

        rem = target - pos
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        if rem < 0:
            # Reducing a long: sell only at/above entry plus edge.
            px = math.ceil(avg_px + EXIT_EDGE_TICKS)
            if best_bid is not None:
                px = max(px, best_bid + 1)
            orders.append(Order(product, int(px), rem))

        elif rem > 0:
            # Reducing a short: buy only at/below entry minus edge.
            px = math.floor(avg_px - EXIT_EDGE_TICKS)
            if best_ask is not None:
                px = min(px, best_ask - 1)
            if px > 0:
                orders.append(Order(product, int(px), rem))

    @staticmethod
    def _market_make_middle(product, pos, fair, od, orders):
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return

        skew = MM_INV_SKEW * pos
        bid_px = int(round(fair - MM_EDGE - skew))
        ask_px = int(round(fair + MM_EDGE - skew))

        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            return

        buy_qty = min(MM_QTY, MM_POS_LIMIT - pos)
        sell_qty = min(MM_QTY, MM_POS_LIMIT + pos)

        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))

    @staticmethod
    def _target_size(z: float) -> int:
        abs_z = abs(z)
        size_mult = 1.0
        if abs_z > RAMP_Z_START:
            ramp = min(1.0, (abs_z - RAMP_Z_START) / (RAMP_Z_FULL - RAMP_Z_START))
            size_mult = 1.0 + ramp * (MAX_SIZE_MULT - 1.0)

        raw_size = K * abs_z * size_mult
        return min(POS_LIMIT, max(1, int(round(raw_size))))

    @staticmethod
    def _entry_threshold(current_sigma: float, sigma_hist) -> float:
        avg_sigma = _mean(sigma_hist) if sigma_hist else current_sigma
        if avg_sigma <= 0:
            vol_ratio = 1.0
        else:
            vol_ratio = current_sigma / avg_sigma

        threshold = ENTRY_Z * vol_ratio
        return max(MIN_ENTRY_Z, min(MAX_ENTRY_Z, threshold))

    def run(self, state: TradingState):
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                pass

        spreads_hist = data.get("spreads", {p: [] for p in TRADE_PRODUCTS})
        vol_hist = data.get("vols", {p: [] for p in TRADE_PRODUCTS})
        for p in TRADE_PRODUCTS:
            spreads_hist.setdefault(p, [])
            vol_hist.setdefault(p, [])

        mids = {}
        for p in ALL_PRODUCTS:
            m = self._mid(state.order_depths.get(p))
            if m is not None:
                mids[p] = m

        result: Dict[str, list] = {p: [] for p in ALL_PRODUCTS}

        def _save_and_return():
            data["spreads"] = spreads_hist
            data["vols"] = vol_hist
            return result, 0, json.dumps(data)

        if any(p not in mids for p in ALL_PRODUCTS):
            return _save_and_return()

        entry_avg = self._reconcile_entry_state(data, state, mids)

        xs = mids["PEBBLES_XS"]
        xl = mids["PEBBLES_XL"]
        anchor = (50000 - xs - xl) / 3

        spreads_now = {}
        for p in TRADE_PRODUCTS:
            spread = mids[p] - anchor
            spreads_now[p] = spread
            spreads_hist[p].append(spread)
            spreads_hist[p] = spreads_hist[p][-WINDOW:]

        if any(len(spreads_hist[p]) < MIN_OBS for p in TRADE_PRODUCTS):
            return _save_and_return()

        zscores = {}
        entry_thresholds = {}
        for p in TRADE_PRODUCTS:
            s = spreads_hist[p]
            sigma = _std(s)
            if sigma == 0:
                return _save_and_return()
            zscores[p] = (spreads_now[p] - _mean(s)) / sigma
            vol_hist[p].append(sigma)
            vol_hist[p] = vol_hist[p][-VOL_WINDOW:]
            entry_thresholds[p] = self._entry_threshold(sigma, vol_hist[p])

        pos = {p: state.position.get(p, 0) for p in TRADE_PRODUCTS}

        ranked = sorted(TRADE_PRODUCTS, key=lambda p: spreads_now[p])
        cheap = ranked[0]
        rich = ranked[-1]

        targets = {}
        for p in TRADE_PRODUCTS:
            z = zscores[p]
            entry_z = entry_thresholds[p]
            cur = pos[p]

            if cur == 0:
                if p == cheap and z <= -entry_z:
                    targets[p] = self._target_size(z)
                elif p == rich and z >= entry_z:
                    targets[p] = -self._target_size(z)
                else:
                    targets[p] = 0
            elif abs(z) < EXIT_Z:
                targets[p] = 0
            elif cur > 0:
                if p != cheap:
                    targets[p] = 0
                elif z <= -entry_z:
                    targets[p] = max(cur, self._target_size(z))
                else:
                    targets[p] = cur
            else:
                if p != rich:
                    targets[p] = 0
                elif z >= entry_z:
                    targets[p] = min(cur, -self._target_size(z))
                else:
                    targets[p] = cur

        for p in TRADE_PRODUCTS:
            target = targets[p]
            cur = pos[p]
            if target == cur:
                continue

            od = state.order_depths.get(p)
            if od is None:
                continue

            is_reducing = (target == 0) or (abs(target) < abs(cur))

            if is_reducing:
                avg_px = entry_avg.get(p, 0.0)
                if self._cross_exit_profitable(cur, avg_px, od):
                    self._cross(p, target, cur, od, result[p])
                else:
                    self._post_profitable_exit(p, target, cur, avg_px, od, result[p])
            else:
                self._post(p, target, cur, od, result[p])

        for p in TRADE_PRODUCTS:
            p_od = state.order_depths.get(p)
            if p_od is not None and not result[p]:
                self._market_make_middle(p, pos[p], anchor, p_od, result[p])

        return _save_and_return()