# velvet_v5.py
#
# Strategy stack:
#   Layer 1 (PRIMARY): Swing reversion — wait for big deviations from fair,
#     take ±200 position, exit when deviation < 3.
#     Same as v3 (which made +$57k backtest, +$9.5k validation).
#   Layer 2 (SECONDARY): Market making during quiet periods.
#     Only active when |target_pos| < QUIET_POS_THRESHOLD AND |position| < same.
#     Posts inside BBO (best_bid+1, best_ask-1) with small size.
#     Captures spread from noise takers (Mark 55) without taking directional risk.
#
# The two layers don't conflict because MM only fires when swing is dormant.
# If a deviation grows past ENTRY_THR, swing takes over and the MM quotes
# get withdrawn (we don't post them at all that tick).

from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass, asdict, field
from typing import List
import json
import math


# ----- Swing strategy params (unchanged from v3) -----
FAIR_ANCHOR = 5247.65
FAIR_DRIFT_CAP = 200.0
FAIR_EMA_ALPHA = 2e-6
ENTRY_THR = 12.0
EXIT_THR = 3.0
MAX_POS_SWING = 200
MIN_HISTORY = 50

# ----- Market-making layer params -----
# MM only active when both target and current position are within this range.
QUIET_POS_THRESHOLD = 30
# Quote size per side. Small enough that even if Mark 67 picks us off, the
# damage is bounded.
MM_QUOTE_SIZE = 30
# How far inside BBO to quote. 1 = standard maker move.
MM_TICKS_INSIDE = 2          # was 1; jump in front of competing makers
# Inventory skew: when we accumulate from MM fills, lean quotes
INVENTORY_SKEW_PER_UNIT = 0.05   # 0.05 ticks per unit of position


@dataclass
class VelvetState:
    fair: float | None = None
    n_obs: int = 0


@dataclass
class SavedState:
    velvet: VelvetState = field(default_factory=lambda: VelvetState())

    @staticmethod
    def load(s: str):
        if not s:
            return SavedState()
        try:
            data = json.loads(s)
            return SavedState(velvet=VelvetState(**data.get("velvet", {})))
        except Exception:
            return SavedState()

    def dump(self):
        return json.dumps(asdict(self))


class Trader:
    VELVET = "VELVETFRUIT_EXTRACT"
    POSITION_LIMIT = 200

    def run(self, state: TradingState):
        saved = SavedState.load(state.traderData)
        orders = {self.VELVET: self.trade_velvet(state, saved)}
        return orders, 0, saved.dump()

    def trade_velvet(self, state: TradingState, saved: SavedState) -> List[Order]:
        product = self.VELVET
        ps = saved.velvet
        orders: List[Order] = []

        depth = state.order_depths.get(product)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return orders

        pos = state.position.get(product, 0)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        if best_bid >= best_ask:
            return orders
        mid = (best_bid + best_ask) / 2.0

        # Update fair (slow EMA, anchored, drift-capped)
        if ps.fair is None:
            ps.fair = FAIR_ANCHOR
        else:
            ps.fair = FAIR_EMA_ALPHA * mid + (1 - FAIR_EMA_ALPHA) * ps.fair
        upper = FAIR_ANCHOR + FAIR_DRIFT_CAP
        lower = FAIR_ANCHOR - FAIR_DRIFT_CAP
        ps.fair = max(lower, min(upper, ps.fair))
        ps.n_obs += 1

        deviation = mid - ps.fair

        if ps.n_obs < MIN_HISTORY:
            return orders

        # ===== LAYER 1: SWING =====
        if deviation > ENTRY_THR:
            swing_target = -MAX_POS_SWING
            swing_regime = "ENTRY_SHORT"
        elif deviation < -ENTRY_THR:
            swing_target = +MAX_POS_SWING
            swing_regime = "ENTRY_LONG"
        elif abs(deviation) < EXIT_THR:
            swing_target = 0
            swing_regime = "EXIT_FLAT"
        else:
            swing_target = pos
            swing_regime = "HOLD"

        # Send swing orders (cross spread to reach target)
        delta = swing_target - pos
        if delta > 0:
            avail = abs(depth.sell_orders[best_ask])
            qty = min(delta, avail)
            if qty > 0:
                orders.append(Order(product, best_ask, int(qty)))
                pos += qty
        elif delta < 0:
            avail = depth.buy_orders[best_bid]
            qty = min(-delta, avail)
            if qty > 0:
                orders.append(Order(product, best_bid, int(-qty)))
                pos -= qty

        # ===== LAYER 2: MARKET MAKING (only when quiet) =====
        # Activate only when:
        #   - Swing isn't trying to position (target near 0)
        #   - Current position is near 0 (we don't have inventory to manage)
        mm_active = (
            abs(swing_target) <= QUIET_POS_THRESHOLD
            and abs(pos) <= QUIET_POS_THRESHOLD
        )

        if mm_active:
            # Quote inside BBO with small size, skewed by inventory
            inv_skew = pos * INVENTORY_SKEW_PER_UNIT  # +ve when long → bias quotes down
            res_price = mid - inv_skew

            # Aggressive: try to be ALONE at our price level (front of queue)
            # by quoting MM_TICKS_INSIDE = 2 when spread is wide enough.
            # Falls back to 1 when spread is tight.
            spread = best_ask - best_bid
            ticks_inside = MM_TICKS_INSIDE if spread >= 4 else 1

            bid_price = min(best_bid + ticks_inside,
                            math.floor(res_price - 1))
            ask_price = max(best_ask - ticks_inside,
                            math.ceil(res_price + 1))

            # Don't cross our own quotes or BBO inside-out
            bid_price = min(bid_price, best_ask - 1)
            ask_price = max(ask_price, best_bid + 1)

            buy_cap = self.POSITION_LIMIT - pos
            sell_cap = self.POSITION_LIMIT + pos
            bid_size = min(MM_QUOTE_SIZE, buy_cap)
            ask_size = min(MM_QUOTE_SIZE, sell_cap)

            if bid_size > 0 and bid_price < ask_price:
                orders.append(Order(product, int(bid_price), int(bid_size)))
            if ask_size > 0 and ask_price > bid_price:
                orders.append(Order(product, int(ask_price), int(-ask_size)))

        # Safety: don't send crossed quotes
        buys = [o for o in orders if o.quantity > 0]
        sells = [o for o in orders if o.quantity < 0]
        if buys and sells:
            max_buy = max(o.price for o in buys)
            min_sell = min(o.price for o in sells)
            if max_buy >= min_sell:
                # Drop any buy whose price is >= a sell's price
                orders = [o for o in orders
                          if not (o.quantity > 0 and o.price >= min_sell)]

        print(f"[VELVET ts={state.timestamp}] mid={mid:.1f} fair={ps.fair:.2f} "
              f"dev={deviation:+.1f} pos={pos} swing={swing_regime} "
              f"swing_target={swing_target} mm={'on' if mm_active else 'off'}")

        return orders