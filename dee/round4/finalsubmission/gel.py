# gel_v4.py - gel_v3.py + Mark 38 detection & exploitation
#
#   Adds on top of gel_v3's 3 surgical changes:
#   4. MARK 38 EXPLOIT: Detect Mark 38 (buys at mid+8, sells at mid-8) via probe orders,
#      then front-run Mark 14 by posting resting quotes at mid±7 once confirmed.
#      Detection uses state.own_trades to validate fills on 1-unit probe orders.
#      Exploitation scales to MARK38_EXPLOIT_QTY=10 per side once confidence≥2.
#      Hard position bias guard: no Mark 38 orders if |pos| > MARK38_POS_BIAS_LIMIT.
#
# Everything else is IDENTICAL to gel_v3.py — same parameters, same logic.

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
    dynamic_mean: float | None = None
    # ── Mark 38 detection state ──────────────────────────────────────────────
    mark38_probe_ask: int | None = None   # ask probe price posted last tick
    mark38_probe_bid: int | None = None   # bid probe price posted last tick
    mark38_confidence: int = 0            # consecutive ticks with confirmed fills
    mark38_miss_streak: int = 0           # consecutive ticks with NO fill (for decay)


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
LONG_MEAN_ANCHOR    = 9994.0
LONG_MEAN_DRIFT_CAP = 60.0
LONG_EMA_ALPHA      = 0.001

LONG_STD  = 32.0

SHORT_LOOKBACK = 400
TREND_LOOKBACK = 50
MAX_HISTORY    = 400

Z_STRONG = 1.5
Z_MILD   = 0.3

MOMENTUM_SCALE              = 30.0
MOMENTUM_MAX_POS_FRACTION   = 0.3
MOMENTUM_REVERSAL_THRESHOLD = 5.0

TREND_DAMPEN_THRESHOLD = 1.5
TREND_FLIP_THRESHOLD   = 3.0
TREND_FOLLOW_FRACTION  = 0.5

GAP_TAKE_THRESHOLD = 80      # was 150 — reach target faster
GAP_NEUTRAL_BAND   = 30

SPREAD_COST_MIN_FRACTION = 0.5   # only take liquidity if expected edge vs dynamic_mean
                                  # exceeds this fraction of the current spread

# ===== CIRCUIT BREAKER =====
DRAWDOWN_THRESHOLD = 1500
LOCKDOWN_DURATION  = 500
LOCKDOWN_POS_CAP   = 30

POSITION_LIMIT = 200
RISK_FACTOR    = 0.05
MAKING_EDGE       = 5    # was 1 — spread is ~15-16 ticks, edge=5 still gets filled
MAKER_CHUNK       = 60   # was full buy_cap/sell_cap — cap individual quote size
MAKER_MAX_RETREAT = 2    # maker quotes never retreat more than this many ticks from best_bid/ask

# ===== MARK 38 EXPLOIT PARAMETERS ============================================
MARK38_OFFSET         = 8     # Mark 38 trades at mid ± this many ticks
MARK38_FRONT_RUN      = 1     # post 1 tick closer to mid than Mark 14 for queue priority
                               #   exploit ask → mid + (OFFSET - FRONT_RUN) = mid + 7
                               #   exploit bid → mid - (OFFSET - FRONT_RUN) = mid - 7
MARK38_PROBE_QTY      = 1     # tiny order to detect Mark 38; fills = confirmation
MARK38_EXPLOIT_QTY    = 10    # order size once Mark 38 is confirmed present
MARK38_MIN_CONFIDENCE = 2     # confirmed fills needed before switching to exploit mode
MARK38_DECAY_TICKS    = 5     # consecutive misses before confidence decrements
MARK38_POS_BIAS_LIMIT = 100   # skip Mark 38 quotes entirely when |pos| > this
                               #   (prevents directional blowup when main strategy is leaning hard)
# =============================================================================


class Trader:
    def __init__(self):
        self.HYDROGEL = "HYDROGEL_PACK"

    def run(self, state: TradingState):
        saved = SavedState.load(state.traderData)
        orders = {}
        orders[self.HYDROGEL] = self.trade_hydrogel(state, saved)
        return orders, 0, saved.dump()

    def update_dynamic_mean(self, ps, mid_price):
        if ps.dynamic_mean is None:
            ps.dynamic_mean = mid_price
        else:
            ps.dynamic_mean = LONG_EMA_ALPHA * mid_price + (1 - LONG_EMA_ALPHA) * ps.dynamic_mean
        upper_bound = LONG_MEAN_ANCHOR + LONG_MEAN_DRIFT_CAP
        lower_bound = LONG_MEAN_ANCHOR - LONG_MEAN_DRIFT_CAP
        ps.dynamic_mean = max(lower_bound, min(upper_bound, ps.dynamic_mean))

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

    def compute_target_position(self, mid_price, history, dynamic_mean):
        z_score = (mid_price - dynamic_mean) / LONG_STD

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

    # =========================================================================
    #  Mark 38 helpers
    # =========================================================================

    def _update_mark38_confidence(self, ps, own_trades, product):
        """
        Inspect own_trades for fills at the probe prices posted last tick.
        Any fill at exactly the stored probe prices counts as a Mark 38 sighting.
        Returns (probe_ask_hit, probe_bid_hit).
        """
        trades = own_trades.get(product, [])
        probe_ask_hit = (
            ps.mark38_probe_ask is not None
            and any(t.price == ps.mark38_probe_ask for t in trades)
        )
        probe_bid_hit = (
            ps.mark38_probe_bid is not None
            and any(t.price == ps.mark38_probe_bid for t in trades)
        )

        if probe_ask_hit or probe_bid_hit:
            ps.mark38_confidence = min(ps.mark38_confidence + 1, 20)
            ps.mark38_miss_streak = 0
        else:
            ps.mark38_miss_streak += 1
            if ps.mark38_miss_streak >= MARK38_DECAY_TICKS:
                ps.mark38_confidence = max(0, ps.mark38_confidence - 1)
                ps.mark38_miss_streak = 0

        return probe_ask_hit, probe_bid_hit

    def _add_mark38_orders(self, ps, product, mid_price, pos, buy_cap, sell_cap, orders):
        """
        Decide whether to probe or exploit, then append orders and record
        the probe prices for next-tick validation.

        Probe mode  (confidence < threshold): post qty=1 at mid ± MARK38_OFFSET
        Exploit mode (confidence ≥ threshold): post qty=MARK38_EXPLOIT_QTY at
                                               mid ± (MARK38_OFFSET - MARK38_FRONT_RUN)
                                               i.e. 1 tick inside Mark 14's quotes.

        Hard guard: skip entirely if |pos| > MARK38_POS_BIAS_LIMIT.
        """
        if abs(pos) > MARK38_POS_BIAS_LIMIT:
            # Position already leaning too hard — don't add inventory pressure.
            ps.mark38_probe_ask = None
            ps.mark38_probe_bid = None
            print(f"MARK38: SKIPPED (|pos|={abs(pos)} > {MARK38_POS_BIAS_LIMIT})")
            return

        exploit_mode = ps.mark38_confidence >= MARK38_MIN_CONFIDENCE

        if exploit_mode:
            # Front-run Mark 14 by 1 tick to guarantee queue priority over them.
            effective_offset = MARK38_OFFSET - MARK38_FRONT_RUN   # = 7
            qty = MARK38_EXPLOIT_QTY
        else:
            # Probe: sit exactly where Mark 38 trades to detect their presence.
            effective_offset = MARK38_OFFSET                       # = 8
            qty = MARK38_PROBE_QTY

        ask_price = int(round(mid_price + effective_offset))
        bid_price = int(round(mid_price - effective_offset))

        # Store for next-tick validation (use probe offset so we know what we posted).
        ps.mark38_probe_ask = ask_price
        ps.mark38_probe_bid = bid_price

        posted_ask = posted_bid = False

        # Post resting ask (sell) — Mark 38 will lift it when they want to buy.
        if sell_cap < 0 and abs(sell_cap) >= qty:
            orders.append(Order(product, ask_price, -qty))
            posted_ask = True

        # Post resting bid (buy) — Mark 38 will hit it when they want to sell.
        if buy_cap > 0 and buy_cap >= qty:
            orders.append(Order(product, bid_price, qty))
            posted_bid = True

        print(
            f"MARK38: mode={'EXPLOIT' if exploit_mode else 'PROBE'} "
            f"conf={ps.mark38_confidence} "
            f"bid@{bid_price}({'✓' if posted_bid else '✗'}) "
            f"ask@{ask_price}({'✓' if posted_ask else '✗'}) "
            f"qty={qty}"
        )

    # =========================================================================

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

        self.update_dynamic_mean(ps, mid_price)

        if ps.last_mid is not None:
            mtm_change = ps.last_pos * (mid_price - ps.last_mid)
            ps.cum_pnl += mtm_change
            ps.peak_pnl = max(ps.peak_pnl, ps.cum_pnl)
        ps.last_mid = mid_price
        ps.last_pos = pos

        drawdown = ps.peak_pnl - ps.cum_pnl
        if not ps.in_lockdown and drawdown > DRAWDOWN_THRESHOLD:
            ps.in_lockdown = True
            ps.lockdown_ticks_remaining = LOCKDOWN_DURATION
            print(f"!!! CIRCUIT BREAKER: dd={drawdown:.0f}")

        if ps.in_lockdown:
            ps.lockdown_ticks_remaining -= 1
            if ps.lockdown_ticks_remaining <= 0:
                ps.in_lockdown = False
                ps.peak_pnl = ps.cum_pnl
                print(f"!!! LOCKDOWN ENDED")

        ps.price_history.append(mid_price)
        if len(ps.price_history) > MAX_HISTORY:
            ps.price_history = ps.price_history[-MAX_HISTORY:]

        # ── Mark 38: validate last tick's probes ─────────────────────────────
        probe_ask_hit, probe_bid_hit = self._update_mark38_confidence(
            ps, state.own_trades, product
        )
        # ─────────────────────────────────────────────────────────────────────

        target_pos, regime, z_score, momentum, trend = self.compute_target_position(
            mid_price, ps.price_history, ps.dynamic_mean
        )

        if ps.in_lockdown:
            if target_pos > LOCKDOWN_POS_CAP:
                target_pos = LOCKDOWN_POS_CAP
            elif target_pos < -LOCKDOWN_POS_CAP:
                target_pos = -LOCKDOWN_POS_CAP
            regime += " | LOCKED"

        gap = target_pos - pos
        buy_cap  = POSITION_LIMIT - pos
        sell_cap = -POSITION_LIMIT - pos

        # ===== POSITION CORRECTION =====
        took_volume = 0
        if abs(gap) > GAP_TAKE_THRESHOLD:
            spread = best_ask - best_bid
            spread_hurdle = spread * SPREAD_COST_MIN_FRACTION
            if gap > 0 and buy_cap > 0:
                take_edge = ps.dynamic_mean - best_ask
                if take_edge >= spread_hurdle:
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
                take_edge = best_bid - ps.dynamic_mean
                if take_edge >= spread_hurdle:
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
            raw_bid  = min(best_bid + 1, math.floor(res_price - MAKING_EDGE))
            bid_price = max(raw_bid, best_bid - MAKER_MAX_RETREAT)
            bid_qty   = min(buy_cap, MAKER_CHUNK)
            orders.append(Order(product, int(bid_price), int(bid_qty)))

        if post_ask and sell_cap < 0:
            raw_ask  = max(best_ask - 1, math.ceil(res_price + MAKING_EDGE))
            ask_price = min(raw_ask, best_ask + MAKER_MAX_RETREAT)
            ask_qty   = max(sell_cap, -MAKER_CHUNK)
            orders.append(Order(product, int(ask_price), int(ask_qty)))

        print(f"mid={mid_price:.1f} dyn_mean={ps.dynamic_mean:.1f} z={z_score:.2f} "
              f"mom={momentum:.1f} trend={trend:.2f} "
              f"regime={regime} target={target_pos} pos_orig={original_pos} took={took_volume} "
              f"gap={gap} cum_pnl={ps.cum_pnl:.0f} peak={ps.peak_pnl:.0f} dd={drawdown:.0f} "
              f"m38_hit=({'A' if probe_ask_hit else '-'}{'B' if probe_bid_hit else '-'})")

        # ===== MARK 38 PROBE / EXPLOIT =====
        # Runs after the main maker so buy_cap/sell_cap reflect what's already committed.
        # Mark 38 orders use the remaining headroom only; they do NOT tighten buy_cap/sell_cap
        # for the safety check below (they're already appended and will be included there).
        self._add_mark38_orders(ps, product, mid_price, pos, buy_cap, sell_cap, orders)

        # ===== SAFETY: cancel crossed orders =====
        buy_orders_list  = [o for o in orders if o.quantity > 0]
        sell_orders_list = [o for o in orders if o.quantity < 0]
        if buy_orders_list and sell_orders_list:
            max_buy  = max(o.price for o in buy_orders_list)
            min_sell = min(o.price for o in sell_orders_list) 
            if max_buy >= min_sell:
                orders = [o for o in orders if not (o.quantity > 0 and o.price >= min_sell)]

        return orders