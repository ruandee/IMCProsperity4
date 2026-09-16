"""
Combined Trader — gel_v4 + velvet_insider + submissionr4 options engine.

Products handled:
  • HYDROGEL_PACK        — mean-reversion / momentum + Mark 38 exploit   (gel_v4)
  • VELVETFRUIT_EXTRACT  — swing + Mark 55 fade                          (velvet_insider)
  • VEV_{5000..5500}     — OU-process options pricing + residual scalp   (submissionr4)

Pipeline per tick:
  1. Load shared SavedState (hydrogel + velvet + options sub-dicts).
  2. Run each sub-strategy independently; each gets its own state slice.
  3. Merge order dicts, persist combined state.

Options pipeline (submissionr4):
  1. OU forward F = mu + (S - mu) * exp(-theta * T)
  2. WLS smile fit (Level, Skew, Convexity) with EWMA-weighted sufficient stats.
  3. Adaptive Kalman update; measurement noise inherits live spread uncertainty.
  4. Wing clamp at ±1.5σ on log-moneyness; linear extrapolation beyond boundary.
  5. Trade signal strikes whose residual exceeds half-spread vega cost.
"""

# ── Shared imports ─────────────────────────────────────────────────────────────
from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass, asdict, field
from typing import Dict, List
import json
import math
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — HYDROGEL constants  (gel_v4)
# ══════════════════════════════════════════════════════════════════════════════
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

GAP_TAKE_THRESHOLD = 80
GAP_NEUTRAL_BAND   = 30

SPREAD_COST_MIN_FRACTION = 0.5

DRAWDOWN_THRESHOLD = 1500
LOCKDOWN_DURATION  = 500
LOCKDOWN_POS_CAP   = 30

HG_POSITION_LIMIT = 200
RISK_FACTOR       = 0.05
MAKING_EDGE       = 5
MAKER_CHUNK       = 60
MAKER_MAX_RETREAT = 2

# Mark 38
MARK38_OFFSET         = 8
MARK38_FRONT_RUN      = 1
MARK38_PROBE_QTY      = 1
MARK38_EXPLOIT_QTY    = 10
MARK38_MIN_CONFIDENCE = 2
MARK38_DECAY_TICKS    = 5
MARK38_POS_BIAS_LIMIT = 100


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — VELVET constants  (velvet_insider)
# ══════════════════════════════════════════════════════════════════════════════
FAIR_ANCHOR    = 5247.65
FAIR_DRIFT_CAP = 200.0
FAIR_EMA_ALPHA = 2e-6
ENTRY_THR      = 12.0
EXIT_THR       = 3.0
MAX_POS_SWING  = 200
MIN_HISTORY    = 50

M55_EDGE             = 2.5
M55_SIZE             = 6
M55_ACTIVE_POS_LIMIT = 130

M55_MIN_QTY             = 3
M55_MAX_QTY             = 8
M55_PROBE_SIZE          = 1
M55_CONFIDENCE_PER_FILL = 2
M55_CONFIDENCE_DECAY    = 1
M55_CONFIDENCE_MAX      = 10
M55_CONFIDENCE_THRESH   = 4

VEL_POSITION_LIMIT = 200


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — OPTIONS constants  (submissionr4)
# ══════════════════════════════════════════════════════════════════════════════
# Seeds (from bootstrap_constants.py)
X0 = np.array([+1.0772156350e-02, +1.0911297429e-01, -1.9502481895e+00])
P0 = np.array([
    [+7.026582e-06, -1.481289e-04, +2.392721e-03],
    [-1.481289e-04, +3.414312e-03, -5.699942e-02],
    [+2.392721e-03, -5.699942e-02, +9.690831e-01],
])
M_STD          = 0.032102
OU_MU          = 5246.8661
OU_THETA       = 21.79498       # per day
TTE_START_DAYS = 5.0
TS_PER_DAY     = 999000

UNDERLYING     = "VELVETFRUIT_EXTRACT"
STRIKES        = [5000, 5100, 5200, 5300, 5400, 5500]
SYMS           = {K: f"VEV_{K}" for K in STRIKES}
SIGNAL_STRIKES = [5100, 5200, 5300, 5400]
VOUCHER_LIMIT  = 300

STALE_TS       = 400
SPREAD_FLOOR   = 0.5
VEGA_FLOOR     = 1e-8
WING_SIGMA     = 1.5
EWMA_ALPHA     = 0.05
WARMUP_NEFF    = 0.30

KF_Q_DIAG  = np.array([1e-7, 1e-7, 1e-8])
KF_R_FLOOR = np.array([1e-6, 1e-6, 1e-7])

CLAMP_LO = -WING_SIGMA * M_STD
CLAMP_HI =  WING_SIGMA * M_STD

SCALP_BASE     = 60
COST_DISCOUNT  = 0.4
NOISE_FLOOR_IV = 0.002

SQRT_2PI = math.sqrt(2.0 * math.pi)
_I3      = np.eye(3)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — State dataclasses
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class HydrogelState:
    price_history:             list       = field(default_factory=list)
    last_mid:                  float|None = None
    last_pos:                  int        = 0
    cum_pnl:                   float      = 0.0
    peak_pnl:                  float      = 0.0
    in_lockdown:               bool       = False
    lockdown_ticks_remaining:  int        = 0
    dynamic_mean:              float|None = None
    mark38_probe_ask:          int|None   = None
    mark38_probe_bid:          int|None   = None
    mark38_confidence:         int        = 0
    mark38_miss_streak:        int        = 0


@dataclass
class VelvetState:
    fair:           float|None = None
    n_obs:          int        = 0
    m55_bought:     int        = 0
    m55_sold:       int        = 0
    m55_cost:       float      = 0.0
    m55_last_bid:   int|None   = None
    m55_last_ask:   int|None   = None
    m55_confidence: int        = 0
    m55_detected:   bool       = False


@dataclass
class SavedState:
    hydrogel: HydrogelState = field(default_factory=HydrogelState)
    velvet:   VelvetState   = field(default_factory=VelvetState)
    options:  dict          = field(default_factory=dict)   # raw dict for options engine

    @staticmethod
    def load(s: str):
        if not s:
            return SavedState()
        try:
            d = json.loads(s)
            return SavedState(
                hydrogel=HydrogelState(**d.get("hydrogel", {})),
                velvet=VelvetState(**d.get("velvet", {})),
                options=d.get("options", {}),
            )
        except Exception:
            return SavedState()

    def dump(self) -> str:
        return json.dumps({
            "hydrogel": asdict(self.hydrogel),
            "velvet":   asdict(self.velvet),
            "options":  self.options,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Options math helpers  (submissionr4)
# ══════════════════════════════════════════════════════════════════════════════
def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _npdf(x): return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_price(F, K, T, sig):
    sq = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / sq
    return F * _ncdf(d1) - K * _ncdf(d1 - sq)


def bs_vega(F, K, T, sig):
    sq = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sig * sig * T) / sq
    return F * _npdf(d1) * math.sqrt(T)


def implied_vol(P, F, K, T, s0):
    if P <= max(F - K, 0.0) + 1e-6:
        return None
    s = s0
    for _ in range(15):
        e = bs_price(F, K, T, s) - P
        if abs(e) < 1e-5:
            return s
        v = bs_vega(F, K, T, s)
        if v < 1e-10:
            break
        s = max(0.0005, min(0.3, s - e / v))
    return s if 0.001 < s < 0.3 else None


def inv3(M):
    """Closed-form 3×3 matrix inverse (faster than np.linalg.solve at this size)."""
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    A =  e * i - f * h
    B = -(d * i - f * g)
    C =  d * h - e * g
    det = a * A + b * B + c * C
    if abs(det) < 1e-18:
        return _I3 * 1e6
    inv_det = 1.0 / det
    out = np.empty((3, 3))
    out[0, 0] = A * inv_det
    out[0, 1] = -(b * i - c * h) * inv_det
    out[0, 2] =  (b * f - c * e) * inv_det
    out[1, 0] = B * inv_det
    out[1, 1] =  (a * i - c * g) * inv_det
    out[1, 2] = -(a * f - c * d) * inv_det
    out[2, 0] = C * inv_det
    out[2, 1] = -(a * h - b * g) * inv_det
    out[2, 2] =  (a * e - b * d) * inv_det
    return out


def _opt_best_bid(od):
    if not od.buy_orders:  return None, 0
    p = max(od.buy_orders); return p, od.buy_orders[p]


def _opt_best_ask(od):
    if not od.sell_orders: return None, 0
    p = min(od.sell_orders); return p, od.sell_orders[p]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Combined Trader
# ══════════════════════════════════════════════════════════════════════════════
class Trader:
    HYDROGEL = "HYDROGEL_PACK"
    VELVET   = "VELVETFRUIT_EXTRACT"

    # ──────────────────────────────────────────────────────────────────────────
    #  Top-level dispatcher
    # ──────────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        saved = SavedState.load(state.traderData)

        result: Dict[str, List[Order]] = {}

        # 1. Hydrogel
        result[self.HYDROGEL] = self._trade_hydrogel(state, saved)

        # 2. Velvet (underlying)
        result[self.VELVET] = self._trade_velvet(state, saved)

        # 3. VEV options
        options_orders, saved.options = self._trade_options(state, saved.options)
        result.update(options_orders)

        return result, 0, saved.dump()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYDROGEL sub-strategy  (gel_v4)
    # ══════════════════════════════════════════════════════════════════════════
    def _trade_hydrogel(self, state: TradingState, saved: SavedState) -> List[Order]:
        product = self.HYDROGEL
        ps      = saved.hydrogel
        orders: List[Order] = []

        if product not in state.order_depths:
            return orders

        order_depth = state.order_depths[product]
        pos          = state.position.get(product, 0)
        original_pos = pos

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid  = max(order_depth.buy_orders.keys())
        best_ask  = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0

        self._hg_update_dynamic_mean(ps, mid_price)

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
            #print(f"!!! CIRCUIT BREAKER: dd={drawdown:.0f}")

        if ps.in_lockdown:
            ps.lockdown_ticks_remaining -= 1
            if ps.lockdown_ticks_remaining <= 0:
                ps.in_lockdown = False
                ps.peak_pnl    = ps.cum_pnl
                #print(f"!!! LOCKDOWN ENDED")

        ps.price_history.append(mid_price)
        if len(ps.price_history) > MAX_HISTORY:
            ps.price_history = ps.price_history[-MAX_HISTORY:]

        # Mark 38: validate last tick's probes
        probe_ask_hit, probe_bid_hit = self._update_mark38_confidence(
            ps, state.own_trades, product
        )

        target_pos, regime, z_score, momentum, trend = self._hg_compute_target(
            mid_price, ps.price_history, ps.dynamic_mean
        )

        if ps.in_lockdown:
            target_pos = max(-LOCKDOWN_POS_CAP, min(LOCKDOWN_POS_CAP, target_pos))
            regime += " | LOCKED"

        gap      = target_pos - pos
        buy_cap  = HG_POSITION_LIMIT - pos
        sell_cap = -HG_POSITION_LIMIT - pos

        # ── Position correction (liquidity take) ──────────────────────────────
        took_volume = 0
        if abs(gap) > GAP_TAKE_THRESHOLD:
            spread        = best_ask - best_bid
            spread_hurdle = spread * SPREAD_COST_MIN_FRACTION
            if gap > 0 and buy_cap > 0:
                take_edge = ps.dynamic_mean - best_ask
                if take_edge >= spread_hurdle:
                    remaining = min(gap, buy_cap)
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        ask_vol = abs(order_depth.sell_orders[ask_price])
                        if remaining <= 0: break
                        fill = min(ask_vol, remaining)
                        if fill > 0:
                            orders.append(Order(product, ask_price, int(fill)))
                            pos += fill; buy_cap -= fill; remaining -= fill
                            took_volume += fill
            elif gap < 0 and sell_cap < 0:
                take_edge = best_bid - ps.dynamic_mean
                if take_edge >= spread_hurdle:
                    remaining = min(abs(gap), abs(sell_cap))
                    for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                        bid_vol = abs(order_depth.buy_orders[bid_price])
                        if remaining <= 0: break
                        fill = min(bid_vol, remaining)
                        if fill > 0:
                            orders.append(Order(product, bid_price, int(-fill)))
                            pos -= fill; sell_cap += fill; remaining -= fill
                            took_volume += fill

        # ── Market making ─────────────────────────────────────────────────────
        gap_after = target_pos - pos
        res_price = mid_price - ((pos - target_pos) * RISK_FACTOR)
        post_bid  = False
        post_ask  = False

        if abs(gap_after) <= GAP_NEUTRAL_BAND:
            post_bid = post_ask = True
        elif gap_after > 0:
            post_bid = True
        else:
            post_ask = True

        if post_bid and buy_cap > 0:
            raw_bid   = min(best_bid + 1, math.floor(res_price - MAKING_EDGE))
            bid_price = max(raw_bid, best_bid - MAKER_MAX_RETREAT)
            bid_qty   = min(buy_cap, MAKER_CHUNK)
            orders.append(Order(product, int(bid_price), int(bid_qty)))

        if post_ask and sell_cap < 0:
            raw_ask   = max(best_ask - 1, math.ceil(res_price + MAKING_EDGE))
            ask_price = min(raw_ask, best_ask + MAKER_MAX_RETREAT)
            ask_qty   = max(sell_cap, -MAKER_CHUNK)
            orders.append(Order(product, int(ask_price), int(ask_qty)))

        # print(f"mid={mid_price:.1f} dyn_mean={ps.dynamic_mean:.1f} z={z_score:.2f} "
        #       f"mom={momentum:.1f} trend={trend:.2f} "
        #       f"regime={regime} target={target_pos} pos_orig={original_pos} took={took_volume} "
        #       f"gap={gap} cum_pnl={ps.cum_pnl:.0f} peak={ps.peak_pnl:.0f} dd={drawdown:.0f} "
        #       f"m38_hit=({'A' if probe_ask_hit else '-'}{'B' if probe_bid_hit else '-'})")

        # ── Mark 38 probe / exploit ───────────────────────────────────────────
        self._add_mark38_orders(ps, product, mid_price, pos, buy_cap, sell_cap, orders)

        # ── Safety: cancel crossed orders ─────────────────────────────────────
        buy_orders_list  = [o for o in orders if o.quantity > 0]
        sell_orders_list = [o for o in orders if o.quantity < 0]
        if buy_orders_list and sell_orders_list:
            max_buy  = max(o.price for o in buy_orders_list)
            min_sell = min(o.price for o in sell_orders_list)
            if max_buy >= min_sell:
                orders = [o for o in orders if not (o.quantity > 0 and o.price >= min_sell)]

        return orders

    # ── Hydrogel helpers ──────────────────────────────────────────────────────
    def _hg_update_dynamic_mean(self, ps: HydrogelState, mid_price: float):
        if ps.dynamic_mean is None:
            ps.dynamic_mean = mid_price
        else:
            ps.dynamic_mean = LONG_EMA_ALPHA * mid_price + (1 - LONG_EMA_ALPHA) * ps.dynamic_mean
        ps.dynamic_mean = max(
            LONG_MEAN_ANCHOR - LONG_MEAN_DRIFT_CAP,
            min(LONG_MEAN_ANCHOR + LONG_MEAN_DRIFT_CAP, ps.dynamic_mean),
        )

    def _hg_compute_trend_strength(self, mid_price: float, history: list) -> float:
        if len(history) < TREND_LOOKBACK:
            return 0.0
        recent        = history[-TREND_LOOKBACK:]
        recent_change = mid_price - recent[0]
        diffs         = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        if not diffs:
            return 0.0
        variance      = sum(d * d for d in diffs) / len(diffs)
        tick_std      = variance ** 0.5
        expected_std  = tick_std * (TREND_LOOKBACK ** 0.5) + 1e-9
        return recent_change / expected_std

    def _hg_compute_target(self, mid_price, history, dynamic_mean):
        z_score = (mid_price - dynamic_mean) / LONG_STD
        momentum = mid_price - history[-SHORT_LOOKBACK] if len(history) >= SHORT_LOOKBACK else 0
        trend_strength = self._hg_compute_trend_strength(mid_price, history)
        abs_trend      = abs(trend_strength)

        if abs(z_score) > Z_MILD:
            wants_long  = (z_score < 0)
            wants_short = (z_score > 0)
            momentum_too_strong = (
                (wants_long  and momentum < -MOMENTUM_REVERSAL_THRESHOLD) or
                (wants_short and momentum >  MOMENTUM_REVERSAL_THRESHOLD)
            )
            if momentum_too_strong:
                baseline_target = 0
                regime = "WAITING_FOR_REVERSAL"
            elif abs(z_score) > Z_STRONG:
                baseline_target = -HG_POSITION_LIMIT * (1 if z_score > 0 else -1)
                regime = "STRONG_REVERSION"
            else:
                scale = (abs(z_score) - Z_MILD) / (Z_STRONG - Z_MILD)
                baseline_target = -int(HG_POSITION_LIMIT * scale * (1 if z_score > 0 else -1))
                regime = "MILD_REVERSION"
        else:
            normalized_momentum = max(-1.0, min(1.0, momentum / MOMENTUM_SCALE))
            baseline_target = int(HG_POSITION_LIMIT * MOMENTUM_MAX_POS_FRACTION * normalized_momentum)
            regime = "MOMENTUM"

        if abs_trend > TREND_FLIP_THRESHOLD:
            target = int(HG_POSITION_LIMIT * TREND_FOLLOW_FRACTION * (1 if trend_strength > 0 else -1))
            regime += " | TREND_FOLLOW"
        elif abs_trend > TREND_DAMPEN_THRESHOLD:
            if (baseline_target > 0) == (trend_strength > 0):
                target = baseline_target
            else:
                dampening = max(
                    0.0,
                    1.0 - (abs_trend - TREND_DAMPEN_THRESHOLD) /
                          (TREND_FLIP_THRESHOLD - TREND_DAMPEN_THRESHOLD),
                )
                target = int(baseline_target * dampening)
                regime += " | DAMPENED"
        else:
            target = baseline_target

        return target, regime, z_score, momentum, trend_strength

    # ── Mark 38 helpers ───────────────────────────────────────────────────────
    def _update_mark38_confidence(self, ps: HydrogelState, own_trades, product: str):
        trades        = own_trades.get(product, [])
        probe_ask_hit = (
            ps.mark38_probe_ask is not None
            and any(t.price == ps.mark38_probe_ask for t in trades)
        )
        probe_bid_hit = (
            ps.mark38_probe_bid is not None
            and any(t.price == ps.mark38_probe_bid for t in trades)
        )
        if probe_ask_hit or probe_bid_hit:
            ps.mark38_confidence  = min(ps.mark38_confidence + 1, 20)
            ps.mark38_miss_streak = 0
        else:
            ps.mark38_miss_streak += 1
            if ps.mark38_miss_streak >= MARK38_DECAY_TICKS:
                ps.mark38_confidence  = max(0, ps.mark38_confidence - 1)
                ps.mark38_miss_streak = 0
        return probe_ask_hit, probe_bid_hit

    def _add_mark38_orders(self, ps: HydrogelState, product, mid_price, pos,
                           buy_cap, sell_cap, orders):
        if abs(pos) > MARK38_POS_BIAS_LIMIT:
            ps.mark38_probe_ask = None
            ps.mark38_probe_bid = None
            #print(f"MARK38: SKIPPED (|pos|={abs(pos)} > {MARK38_POS_BIAS_LIMIT})")
            return

        exploit_mode = ps.mark38_confidence >= MARK38_MIN_CONFIDENCE
        if exploit_mode:
            effective_offset = MARK38_OFFSET - MARK38_FRONT_RUN   # = 7
            qty = MARK38_EXPLOIT_QTY
        else:
            effective_offset = MARK38_OFFSET                       # = 8
            qty = MARK38_PROBE_QTY

        ask_price = int(round(mid_price + effective_offset))
        bid_price = int(round(mid_price - effective_offset))

        ps.mark38_probe_ask = ask_price
        ps.mark38_probe_bid = bid_price

        posted_ask = posted_bid = False

        if sell_cap < 0 and abs(sell_cap) >= qty:
            orders.append(Order(product, ask_price, -qty))
            posted_ask = True

        if buy_cap > 0 and buy_cap >= qty:
            orders.append(Order(product, bid_price, qty))
            posted_bid = True

        # print(
        #     f"MARK38: mode={'EXPLOIT' if exploit_mode else 'PROBE'} "
        #     f"conf={ps.mark38_confidence} "
        #     f"bid@{bid_price}({'✓' if posted_bid else '✗'}) "
        #     f"ask@{ask_price}({'✓' if posted_ask else '✗'}) "
        #     f"qty={qty}"
        # )

    # ══════════════════════════════════════════════════════════════════════════
    #  VELVET sub-strategy  (velvet_insider)
    # ══════════════════════════════════════════════════════════════════════════
    def _trade_velvet(self, state: TradingState, saved: SavedState) -> List[Order]:
        ps     = saved.velvet
        orders: List[Order] = []

        # ── STEP 1: tally M55 fills from last tick via own_trades ──────────────
        got_m55_fill = False
        for t in state.own_trades.get(self.VELVET, []):
            qty = t.quantity
            if (
                ps.m55_last_bid is not None
                and t.price == ps.m55_last_bid
                and qty > 0
                and M55_MIN_QTY <= qty <= M55_MAX_QTY
            ):
                ps.m55_bought += qty
                ps.m55_cost   += t.price * qty
                got_m55_fill   = True
            elif (
                ps.m55_last_ask is not None
                and t.price == ps.m55_last_ask
                and qty < 0
                and M55_MIN_QTY <= abs(qty) <= M55_MAX_QTY
            ):
                ps.m55_sold += abs(qty)
                ps.m55_cost -= t.price * abs(qty)
                got_m55_fill  = True

        # ── STEP 2: update confidence score ────────────────────────────────────
        if got_m55_fill:
            ps.m55_confidence = min(M55_CONFIDENCE_MAX,
                                    ps.m55_confidence + M55_CONFIDENCE_PER_FILL)
        else:
            ps.m55_confidence = max(0, ps.m55_confidence - M55_CONFIDENCE_DECAY)
        ps.m55_detected = ps.m55_confidence >= M55_CONFIDENCE_THRESH

        # ── Market data ────────────────────────────────────────────────────────
        depth = state.order_depths.get(self.VELVET)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return orders

        pos      = state.position.get(self.VELVET, 0)
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        if best_bid >= best_ask:
            return orders
        mid = (best_bid + best_ask) / 2.0

        # ── Fair value EMA ─────────────────────────────────────────────────────
        if ps.fair is None:
            ps.fair = FAIR_ANCHOR
        else:
            ps.fair = FAIR_EMA_ALPHA * mid + (1 - FAIR_EMA_ALPHA) * ps.fair
        ps.fair = max(FAIR_ANCHOR - FAIR_DRIFT_CAP,
                      min(FAIR_ANCHOR + FAIR_DRIFT_CAP, ps.fair))
        ps.n_obs += 1

        if ps.n_obs < MIN_HISTORY:
            return orders

        dev = mid - ps.fair

        # ── LAYER 1: swing ─────────────────────────────────────────────────────
        if   dev >  ENTRY_THR: swing_target = -MAX_POS_SWING
        elif dev < -ENTRY_THR: swing_target = +MAX_POS_SWING
        elif abs(dev) < EXIT_THR: swing_target = 0
        else: swing_target = 0

        swing_sent = False
        delta      = swing_target - pos
        is_entry   = abs(swing_target) == MAX_POS_SWING

        if delta > 0:
            if is_entry:
                qty = min(delta, abs(depth.sell_orders[best_ask]))
                if qty > 0:
                    orders.append(Order(self.VELVET, best_ask, int(qty)))
                    pos += qty; swing_sent = True
            elif pos < 0:
                orders.append(Order(self.VELVET, best_bid, delta))
                pos += delta; swing_sent = True
        elif delta < 0:
            if is_entry:
                qty = min(-delta, depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(self.VELVET, best_bid, int(-qty)))
                    pos -= qty; swing_sent = True
            elif pos > 0:
                orders.append(Order(self.VELVET, best_ask, delta))
                pos += delta; swing_sent = True

        # ── LAYER 2: Mark 55 fade (detection-gated) ────────────────────────────
        swing_dormant = abs(swing_target) != MAX_POS_SWING
        m55_layer_ok  = (
            not swing_sent
            and swing_dormant
            and abs(pos) <= M55_ACTIVE_POS_LIMIT
        )

        if m55_layer_ok:
            bid_p    = round(mid - M55_EDGE)
            ask_p    = round(mid + M55_EDGE)
            buy_cap  = VEL_POSITION_LIMIT - pos
            sell_cap = VEL_POSITION_LIMIT + pos

            target_size = M55_SIZE if ps.m55_detected else M55_PROBE_SIZE

            bid_sz = min(target_size, buy_cap)
            ask_sz = min(target_size, sell_cap)

            if bid_sz > 0:
                orders.append(Order(self.VELVET, bid_p,  int(bid_sz)))
            if ask_sz > 0:
                orders.append(Order(self.VELVET, ask_p, -int(ask_sz)))

            ps.m55_last_bid = bid_p if bid_sz > 0 else None
            ps.m55_last_ask = ask_p if ask_sz > 0 else None
        else:
            ps.m55_last_bid = None
            ps.m55_last_ask = None

        # ── End-of-day M55 tally ───────────────────────────────────────────────
        if state.timestamp == 999000:
            net_inv   = ps.m55_bought - ps.m55_sold
            mtm_value = net_inv * mid
            realised  = -ps.m55_cost
            # total_pnl = realised + mtm_value

        return orders

    # ══════════════════════════════════════════════════════════════════════════
    #  OPTIONS sub-strategy  (submissionr4)
    # ══════════════════════════════════════════════════════════════════════════
    def _trade_options(self, state: TradingState, ps: dict):
        """
        Returns (orders_dict, new_options_state_dict).
        """
        result: Dict[str, List[Order]] = {sym: [] for sym in SYMS.values()}

        # ── Restore options state ─────────────────────────────────────────────
        x      = np.array(ps.get("x",  X0.tolist()),            dtype=float)
        P      = np.array(ps.get("P",  P0.flatten().tolist()),  dtype=float).reshape(3, 3)
        S_AA   = np.array(ps.get("AA", [0.0] * 9),              dtype=float).reshape(3, 3)
        S_Ay   = np.array(ps.get("Ay", [0.0] * 3),              dtype=float)
        n_eff  = float(ps.get("ne", 0.0))
        ivs    = ps.get("ivs", {str(K): 0.015 for K in STRIKES})
        last_q = ps.get("q",  {})
        last_t = ps.get("lts", {})

        # ── Time / spot / OU forward ──────────────────────────────────────────
        tick   = int(state.timestamp)
        T_days = max(TTE_START_DAYS - tick / TS_PER_DAY, 1e-9)

        S_obs = None
        if UNDERLYING in state.order_depths:
            od = state.order_depths[UNDERLYING]
            bp, _ = _opt_best_bid(od)
            ap, _ = _opt_best_ask(od)
            if bp is not None and ap is not None:
                S_obs = 0.5 * (bp + ap)
        if S_obs is None:
            return result, ps

        F = OU_MU + (S_obs - OU_MU) * math.exp(-OU_THETA * T_days)

        # ── Per-strike: IV, vega, spread, freshness ───────────────────────────
        K_a, iv_a, v_a, sp_a, fr_a = [], [], [], [], []
        live_bids: Dict[int, tuple] = {}
        live_asks: Dict[int, tuple] = {}

        for K in STRIKES:
            sym = SYMS[K]
            if sym not in state.order_depths: continue
            od = state.order_depths[sym]
            bp, bq = _opt_best_bid(od)
            ap, aq = _opt_best_ask(od)
            if bp is None or ap is None: continue
            mid    = 0.5 * (bp + ap)
            spread = max(ap - bp, SPREAD_FLOOR)

            seed = float(ivs.get(str(K), 0.015))
            iv   = implied_vol(mid, F, K, T_days, seed)
            if iv is None: continue
            ivs[str(K)] = iv

            v  = max(bs_vega(F, K, T_days, iv), VEGA_FLOOR)
            kk = str(K)
            cur = [bp, ap]
            if last_q.get(kk) != cur:
                last_q[kk] = cur
                last_t[kk] = tick
            age   = tick - int(last_t.get(kk, tick))
            fresh = 1.0 if age <= STALE_TS else 0.0

            K_a.append(K);   iv_a.append(iv);     v_a.append(v)
            sp_a.append(spread); fr_a.append(fresh)
            live_bids[K] = (bp, bq)
            live_asks[K] = (ap, aq)

        # ── EWMA-update sufficient stats ──────────────────────────────────────
        m_np = None
        if len(K_a) >= 3:
            K_np  = np.asarray(K_a,  dtype=float)
            iv_np = np.asarray(iv_a, dtype=float)
            v_np  = np.asarray(v_a,  dtype=float)
            sp_np = np.asarray(sp_a, dtype=float)
            fr_np = np.asarray(fr_a, dtype=float)

            m_np  = np.log(K_np / F)
            w_np  = (v_np / v_np.max()) / sp_np * fr_np

            X    = np.column_stack([np.ones_like(m_np), m_np, m_np * m_np])
            Xw   = X * w_np[:, None]
            AA   = Xw.T @ X
            Ay   = Xw.T @ iv_np

            S_AA = (1.0 - EWMA_ALPHA) * S_AA + EWMA_ALPHA * AA
            S_Ay = (1.0 - EWMA_ALPHA) * S_Ay + EWMA_ALPHA * Ay
            n_eff = (1.0 - EWMA_ALPHA) * n_eff + EWMA_ALPHA * (1.0 if w_np.any() else 0.0)

        # ── WLS solve + adaptive Kalman update ────────────────────────────────
        if n_eff > WARMUP_NEFF and m_np is not None:
            cov_raw  = inv3(S_AA)
            beta_raw = cov_raw @ S_Ay

            R_diag = np.maximum(np.diag(cov_raw), KF_R_FLOOR)
            R      = np.diag(R_diag)

            P_pred = P + np.diag(KF_Q_DIAG)
            S_inn  = P_pred + R
            K_g    = P_pred @ inv3(S_inn)
            x      = x + K_g @ (beta_raw - x)

            I_KH = _I3 - K_g
            P    = I_KH @ P_pred @ I_KH.T + K_g @ R @ K_g.T
        else:
            P = P + np.diag(KF_Q_DIAG)

        # ── Wing-clamped fitted IV ─────────────────────────────────────────────
        c0, c1, c2 = float(x[0]), float(x[1]), float(x[2])
        sig_lo   = c0 + c1 * CLAMP_LO + c2 * CLAMP_LO * CLAMP_LO
        sig_hi   = c0 + c1 * CLAMP_HI + c2 * CLAMP_HI * CLAMP_HI
        slope_lo = c1 + 2.0 * c2 * CLAMP_LO
        slope_hi = c1 + 2.0 * c2 * CLAMP_HI

        def fit_iv(m):
            if m < CLAMP_LO: return sig_lo + slope_lo * (m - CLAMP_LO)
            if m > CLAMP_HI: return sig_hi + slope_hi * (m - CLAMP_HI)
            return c0 + c1 * m + c2 * m * m

        # ── Trade signal strikes on residual edge ─────────────────────────────
        if m_np is not None and n_eff > WARMUP_NEFF:
            v_max = float(np.asarray(v_a).max())
            for K, iv, m, v, spread in zip(K_a, iv_a, m_np, v_a, sp_a):
                iv_f = fit_iv(float(m))
                r    = iv - iv_f
                if K not in SIGNAL_STRIKES: continue

                cost_iv = (spread * 0.5) / v
                thresh  = max(cost_iv * COST_DISCOUNT, NOISE_FLOOR_IV)
                if abs(r) < thresh: continue

                sym = SYMS[K]
                pos = state.position.get(sym, 0)
                bp, bq = live_bids[K]
                ap, aq = live_asks[K]

                size = int(round(SCALP_BASE * (v / v_max)))
                if size <= 0: continue

                if r < 0:                           # CHEAP → BUY at ask
                    room = VOUCHER_LIMIT - pos
                    qty  = min(size, room, max(-aq, 0))
                    if qty > 0:
                        result[sym].append(Order(sym, ap, qty))
                else:                               # RICH → SELL at bid
                    room = VOUCHER_LIMIT + pos
                    qty  = min(size, room, max(bq, 0))
                    if qty > 0:
                        result[sym].append(Order(sym, bp, -qty))

        # ── Position-limit safety ──────────────────────────────────────────────
        for K in STRIKES:
            sym = SYMS[K]
            pos = state.position.get(sym, 0)
            tb  = sum(o.quantity  for o in result[sym] if o.quantity > 0)
            ts  = sum(-o.quantity for o in result[sym] if o.quantity < 0)
            if pos + tb > VOUCHER_LIMIT or pos - ts < -VOUCHER_LIMIT:
                result[sym] = []

        # ── Persist options state ─────────────────────────────────────────────
        new_ps = {
            "x":   x.tolist(),
            "P":   P.flatten().tolist(),
            "AA":  S_AA.flatten().tolist(),
            "Ay":  S_Ay.tolist(),
            "ne":  n_eff,
            "ivs": ivs,
            "q":   last_q,
            "lts": last_t,
        }
        return result, new_ps