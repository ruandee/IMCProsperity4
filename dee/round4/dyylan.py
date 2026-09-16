from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass, asdict, field
from typing import List
import json
import math


@dataclass
class HydrogelState:
    price_history: list = field(default_factory=list)
    last_mid: float | None = None
    last_pos: int = 0
    cum_pnl: float = 0.0
    peak_pnl: float = 0.0
    in_lockdown: bool = False
    lockdown_ticks_remaining: int = 0


@dataclass
class SavedState:
    hydrogel: HydrogelState = field(default_factory=lambda: HydrogelState())

    @staticmethod
    def load(s: str):
        if not s:
            return SavedState()
        data = json.loads(s)
        return SavedState(
            hydrogel=HydrogelState(**data.get("hydrogel", {})),
        )

    def dump(self):
        return json.dumps(asdict(self))


# ===== STRATEGY PARAMETERS =====
LONG_MEAN = 9991.0
LONG_STD  = 32.0

SHORT_LOOKBACK = 400
TREND_LOOKBACK = 50
MAX_HISTORY    = 400  # 40.py setting (momentum broken)

Z_STRONG = 1.5
Z_MILD = 0.3

MOMENTUM_SCALE = 30.0
MOMENTUM_MAX_POS_FRACTION = 0.3
MOMENTUM_REVERSAL_THRESHOLD = 5.0

TREND_DAMPEN_THRESHOLD = 1.5
TREND_FLIP_THRESHOLD   = 3.0
TREND_FOLLOW_FRACTION  = 0.5

GAP_TAKE_THRESHOLD = 150
GAP_NEUTRAL_BAND   = 30

# ===== CIRCUIT BREAKER =====
DRAWDOWN_THRESHOLD = 1500       # if cum_pnl drops by this from peak → lockdown
LOCKDOWN_DURATION = 500         # how many ticks of reduced activity
LOCKDOWN_POS_CAP = 30           # max abs position during lockdown
# ===========================

POSITION_LIMIT = 200
RISK_FACTOR    = 0.05
MAKING_EDGE    = 1
# ================================


class Trader:
    def __init__(self):
        self.HYDROGEL = "HYDROGEL_PACK"

    def run(self, state: TradingState):
        saved = SavedState.load(state.traderData)
        orders = {}
        orders[self.HYDROGEL] = self.trade_hydrogel(state, saved)
        return orders, 0, saved.dump()

    def compute_trend_strength(self, mid_price, history):
        if len(history) < TREND_LOOKBACK:
            return 0.0
        recent = history[-TREND_LOOKBACK:]
        recent_change = mid_price - recent[0]
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        if not diffs:
            return 0.0
        variance = sum(d * d for d in diffs) / len(diffs)
        tick_std = variance ** 0.5
        expected_std = tick_std * (TREND_LOOKBACK ** 0.5) + 1e-9
        return recent_change / expected_std

    def compute_target_position(self, mid_price, history):
        z_score = (mid_price - LONG_MEAN) / LONG_STD

        if len(history) >= SHORT_LOOKBACK:
            momentum = mid_price - history[-SHORT_LOOKBACK]
        else:
            momentum = 0

        trend_strength = self.compute_trend_strength(mid_price, history)
        abs_trend = abs(trend_strength)

        if abs(z_score) > Z_MILD:
            wants_long = (z_score < 0)
            wants_short = (z_score > 0)
            momentum_too_strong = (
                (wants_long and momentum < -MOMENTUM_REVERSAL_THRESHOLD) or
                (wants_short and momentum > MOMENTUM_REVERSAL_THRESHOLD)
            )
            if momentum_too_strong:
                baseline_target = 0
                regime = "WAITING_FOR_REVERSAL"
            elif abs(z_score) > Z_STRONG:
                baseline_target = -POSITION_LIMIT * (1 if z_score > 0 else -1)
                regime = "STRONG_REVERSION"
            else:
                scale = (abs(z_score) - Z_MILD) / (Z_STRONG - Z_MILD)
                baseline_target = -int(POSITION_LIMIT * scale * (1 if z_score > 0 else -1))
                regime = "MILD_REVERSION"
        else:
            normalized_momentum = max(-1.0, min(1.0, momentum / MOMENTUM_SCALE))
            baseline_target = int(POSITION_LIMIT * MOMENTUM_MAX_POS_FRACTION * normalized_momentum)
            regime = "MOMENTUM"

        if abs_trend > TREND_FLIP_THRESHOLD:
            target = int(POSITION_LIMIT * TREND_FOLLOW_FRACTION * (1 if trend_strength > 0 else -1))
            regime += " | TREND_FOLLOW"
        elif abs_trend > TREND_DAMPEN_THRESHOLD:
            if (baseline_target > 0) == (trend_strength > 0):
                target = baseline_target
            else:
                dampening = max(0.0, 1.0 - (abs_trend - TREND_DAMPEN_THRESHOLD) /
                                          (TREND_FLIP_THRESHOLD - TREND_DAMPEN_THRESHOLD))
                target = int(baseline_target * dampening)
                regime += " | DAMPENED"
        else:
            target = baseline_target

        return target, regime, z_score, momentum, trend_strength

    def trade_hydrogel(self, state: TradingState, saved: SavedState) -> List[Order]:
        product = self.HYDROGEL
        ps = saved.hydrogel
        orders: List[Order] = []

        if product not in state.order_depths:
            return orders

        order_depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        original_pos = pos

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0

        # ===== UPDATE CUMULATIVE PnL TRACKING =====
        if ps.last_mid is not None:
            mtm_change = ps.last_pos * (mid_price - ps.last_mid)
            ps.cum_pnl += mtm_change
            ps.peak_pnl = max(ps.peak_pnl, ps.cum_pnl)
        ps.last_mid = mid_price
        ps.last_pos = pos

        # ===== CIRCUIT BREAKER LOGIC =====
        drawdown = ps.peak_pnl - ps.cum_pnl
        if not ps.in_lockdown and drawdown > DRAWDOWN_THRESHOLD:
            ps.in_lockdown = True
            ps.lockdown_ticks_remaining = LOCKDOWN_DURATION
            print(f"!!! CIRCUIT BREAKER: drawdown={drawdown:.0f} from peak={ps.peak_pnl:.0f} -> LOCKDOWN")

        if ps.in_lockdown:
            ps.lockdown_ticks_remaining -= 1
            if ps.lockdown_ticks_remaining <= 0:
                ps.in_lockdown = False
                ps.peak_pnl = ps.cum_pnl  # reset peak after lockdown
                print(f"!!! LOCKDOWN ENDED, resetting peak")
        # ===================================

        ps.price_history.append(mid_price)
        if len(ps.price_history) > MAX_HISTORY:
            ps.price_history = ps.price_history[-MAX_HISTORY:]

        target_pos, regime, z_score, momentum, trend = self.compute_target_position(
            mid_price, ps.price_history
        )

        # During lockdown, cap target position
        if ps.in_lockdown:
            if target_pos > LOCKDOWN_POS_CAP:
                target_pos = LOCKDOWN_POS_CAP
            elif target_pos < -LOCKDOWN_POS_CAP:
                target_pos = -LOCKDOWN_POS_CAP
            regime += " | LOCKED"

        gap = target_pos - pos
        buy_cap = POSITION_LIMIT - pos
        sell_cap = -POSITION_LIMIT - pos

        # ===== POSITION CORRECTION =====
        took_volume = 0
        if abs(gap) > GAP_TAKE_THRESHOLD:
            if gap > 0 and buy_cap > 0:
                remaining = min(gap, buy_cap)
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    ask_vol = abs(order_depth.sell_orders[ask_price])
                    if remaining <= 0:
                        break
                    fill = min(ask_vol, remaining)
                    if fill > 0:
                        orders.append(Order(product, ask_price, int(fill)))
                        pos += fill
                        buy_cap -= fill
                        remaining -= fill
                        took_volume += fill
            elif gap < 0 and sell_cap < 0:
                remaining = min(abs(gap), abs(sell_cap))
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    bid_vol = abs(order_depth.buy_orders[bid_price])
                    if remaining <= 0:
                        break
                    fill = min(bid_vol, remaining)
                    if fill > 0:
                        orders.append(Order(product, bid_price, int(-fill)))
                        pos -= fill
                        sell_cap += fill
                        remaining -= fill
                        took_volume += fill

        # ===== MAKE =====
        gap_after = target_pos - pos
        res_price = mid_price - ((pos - target_pos) * RISK_FACTOR)

        post_bid = False
        post_ask = False

        if abs(gap_after) <= GAP_NEUTRAL_BAND:
            post_bid = True
            post_ask = True
        elif gap_after > 0:
            post_bid = True
        else:
            post_ask = True

        if post_bid and buy_cap > 0:
            bid_price = min(best_bid + 1, math.floor(res_price - MAKING_EDGE))
            orders.append(Order(product, int(bid_price), int(buy_cap)))

        if post_ask and sell_cap < 0:
            ask_price = max(best_ask - 1, math.ceil(res_price + MAKING_EDGE))
            orders.append(Order(product, int(ask_price), int(sell_cap)))

        # print(f"mid={mid_price:.1f} z={z_score:.2f} mom={momentum:.1f} trend={trend:.2f} "
        #       f"regime={regime} target={target_pos} pos_orig={original_pos} took={took_volume} "
        #       f"gap={gap} post_bid={post_bid} post_ask={post_ask} "
        #       f"cum_pnl={ps.cum_pnl:.0f} peak={ps.peak_pnl:.0f} dd={drawdown:.0f}")

        # Safety
        buy_orders_list = [o for o in orders if o.quantity > 0]
        sell_orders_list = [o for o in orders if o.quantity < 0]
        if buy_orders_list and sell_orders_list:
            max_buy = max(o.price for o in buy_orders_list)
            min_sell = min(o.price for o in sell_orders_list)
            if max_buy >= min_sell:
                orders = [o for o in orders if not (o.quantity > 0 and o.price >= min_sell)]

        return orders