"""
snacks_v4.py — Round 5 multi-strategy trader (v4 of snackpack basket strategy)

Changes from v3:
  1. Corrected priors. v3 used CHOC-VAN prior_mean=-100, but the actual
     measured mean is -254 (CHOC ~9843, VAN ~10097). v3 took ~3000 ticks
     to drift to true value, costing us PnL. v4 uses measured values.
       - CHOC-VAN: mean=-254, std=65 (true σ of the difference)
       - STRAW-RASP: mean=629, std=200
  2. Faster STRAW-RASP EMA (1/500 → 1/300) since the regime drifts.
  3. NEW: Standalone PIS (Pistachio) mean reverter.
       - PIS doesn't have a stable partner relationship across days
       - But it DOES mean-revert intraday (half-life ~850-1250 ticks)
       - It drifts day-to-day (mean: 9655 → 9487 → 9344, ~-$100/day)
       - Strategy: rolling EMA z-score on PIS alone, same hard-cap framework
  4. PIS uses its own Strategy class. Modular — same interface.

PnL expectation: CHOC/VAN/STRAW/RASP ≈ same as v3. PIS is a small bonus
($1k-3k per 3 days plausibly, given ~$130 daily std).
"""

from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import json
import math


POSITION_LIMIT = 10


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class SavedState:
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data or {}

    @staticmethod
    def load(s: str) -> "SavedState":
        if not s:
            return SavedState()
        try:
            return SavedState(json.loads(s))
        except Exception:
            return SavedState()

    def get(self, key: str) -> Dict[str, Any]:
        return self.data.setdefault(key, {})

    def put(self, key: str, value: Dict[str, Any]):
        self.data[key] = value

    def dump(self) -> str:
        return json.dumps(self.data)


# ============================================================================
# UTILITIES
# ============================================================================

def best_bid_ask_mid(order_depth: OrderDepth):
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None, None, None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return best_bid, best_ask, (best_bid + best_ask) / 2.0


def order_book_imbalance_l1(order_depth: OrderDepth) -> float:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    bv = order_depth.buy_orders[best_bid]
    av = abs(order_depth.sell_orders[best_ask])
    denom = bv + av
    return (bv - av) / denom if denom > 0 else 0.0


def take_liquidity_buy(product: str, order_depth: OrderDepth,
                       max_qty: int) -> List[Order]:
    orders = []
    remaining = max_qty
    for ask_price in sorted(order_depth.sell_orders.keys()):
        if remaining <= 0:
            break
        ask_vol = abs(order_depth.sell_orders[ask_price])
        fill = min(ask_vol, remaining)
        if fill > 0:
            orders.append(Order(product, ask_price, int(fill)))
            remaining -= fill
    return orders


def take_liquidity_sell(product: str, order_depth: OrderDepth,
                        max_qty: int) -> List[Order]:
    orders = []
    remaining = max_qty
    for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
        if remaining <= 0:
            break
        bid_vol = order_depth.buy_orders[bid_price]
        fill = min(bid_vol, remaining)
        if fill > 0:
            orders.append(Order(product, bid_price, int(-fill)))
            remaining -= fill
    return orders


# ============================================================================
# Shared helpers for EMA-based mean reversion
# ============================================================================

def update_ema(state: dict, value: float, alpha: float):
    """Online EMA mean+variance update. Initializes on first call."""
    if not state.get("initialized", False):
        state["ema_mean"] = value
        state["ema_var"] = state.get("ema_var", 1.0)
        state["initialized"] = True
        state["n_ticks"] = state.get("n_ticks", 0) + 1
        state["last_value"] = value
        return
    old_mean = state["ema_mean"]
    diff = value - old_mean
    state["ema_var"] = (1 - alpha) * (state["ema_var"] + alpha * diff * diff)
    state["ema_mean"] = old_mean + alpha * diff
    state["n_ticks"] += 1
    state["last_value"] = value


def compute_z_blended(state: dict, prior_mean: float, prior_std: float,
                      alpha: float) -> float:
    """Z-score with prior blending during warmup."""
    std_emp = math.sqrt(max(state["ema_var"], 1.0))
    warmup_target = 3.0 / alpha
    blend = min(1.0, state["n_ticks"] / warmup_target)
    std = blend * std_emp + (1 - blend) * prior_std
    mean = blend * state["ema_mean"] + (1 - blend) * prior_mean
    if std <= 0:
        return 0.0
    return (state["last_value"] - mean) / std


def z_to_magnitude(z: float, z_min: float, z_full: float) -> int:
    """|z|-to-position-magnitude in [0, POSITION_LIMIT]."""
    abs_z = abs(z)
    if abs_z <= z_min:
        return 0
    if abs_z >= z_full:
        return POSITION_LIMIT
    scale = (abs_z - z_min) / (z_full - z_min)
    return int(round(POSITION_LIMIT * scale))


def execute_leg_capped(product: str, depth: OrderDepth,
                       best_bid: int, best_ask: int,
                       target_pos: int, current_pos: int,
                       cross_allowed: bool,
                       my_orders: Dict[str, List[Order]],
                       mm_base_size: int = 1,
                       mm_max_deviation: int = 4,
                       cross_min_gap: int = 5):
    """Single-leg order placement with hard position caps and MM-at-target.
    Produces at most one buy and one sell per call. Used by both spread
    strategy and standalone PIS strategy.
    """
    buy_cap = max(0, POSITION_LIMIT - current_pos)
    sell_cap = max(0, POSITION_LIMIT + current_pos)

    gap = target_pos - current_pos

    bid_price = best_bid + 1 if (best_ask - best_bid) > 1 else best_bid
    ask_price = best_ask - 1 if (best_ask - best_bid) > 1 else best_ask

    my_buy_qty = 0
    my_sell_qty = 0

    if gap > 0:
        qty_wanted = min(gap, buy_cap)
        if cross_allowed and abs(gap) >= cross_min_gap and qty_wanted > 0:
            taken = take_liquidity_buy(product, depth, qty_wanted)
            taken_qty = sum(o.quantity for o in taken)
            if taken_qty > buy_cap:
                cap_remaining = buy_cap
                fixed = []
                for o in taken:
                    if cap_remaining <= 0:
                        break
                    q = min(o.quantity, cap_remaining)
                    fixed.append(Order(product, o.price, int(q)))
                    cap_remaining -= q
                taken = fixed
                taken_qty = sum(o.quantity for o in taken)
            my_orders.setdefault(product, []).extend(taken)
            buy_cap -= taken_qty
            qty_wanted -= taken_qty
        if qty_wanted > 0 and buy_cap > 0:
            my_buy_qty = min(qty_wanted, buy_cap)
            buy_cap -= my_buy_qty
    elif gap < 0:
        qty_wanted = min(-gap, sell_cap)
        if cross_allowed and abs(gap) >= cross_min_gap and qty_wanted > 0:
            taken = take_liquidity_sell(product, depth, qty_wanted)
            taken_qty = sum(-o.quantity for o in taken)
            if taken_qty > sell_cap:
                cap_remaining = sell_cap
                fixed = []
                for o in taken:
                    if cap_remaining <= 0:
                        break
                    q = min(-o.quantity, cap_remaining)
                    fixed.append(Order(product, o.price, int(-q)))
                    cap_remaining -= q
                taken = fixed
                taken_qty = sum(-o.quantity for o in taken)
            my_orders.setdefault(product, []).extend(taken)
            sell_cap -= taken_qty
            qty_wanted -= taken_qty
        if qty_wanted > 0 and sell_cap > 0:
            my_sell_qty = min(qty_wanted, sell_cap)
            sell_cap -= my_sell_qty

    # MM-at-target: add passive quotes that won't push past target±MM_MAX_DEVIATION
    target_drift_max = target_pos + mm_max_deviation
    target_drift_min = target_pos - mm_max_deviation

    mm_buy_room = target_drift_max - current_pos - my_buy_qty
    mm_buy_qty = min(mm_base_size, max(0, mm_buy_room), buy_cap)

    mm_sell_room = (current_pos - my_sell_qty) - target_drift_min
    mm_sell_qty = min(mm_base_size, max(0, mm_sell_room), sell_cap)

    total_buy_qty = my_buy_qty + mm_buy_qty
    total_sell_qty = my_sell_qty + mm_sell_qty

    max_total_buy = POSITION_LIMIT - current_pos
    if total_buy_qty > max_total_buy:
        total_buy_qty = max(0, max_total_buy)
    max_total_sell = POSITION_LIMIT + current_pos
    if total_sell_qty > max_total_sell:
        total_sell_qty = max(0, max_total_sell)

    if total_buy_qty > 0 and total_sell_qty > 0:
        if bid_price >= ask_price:
            if my_buy_qty > my_sell_qty:
                total_sell_qty = 0
            else:
                total_buy_qty = 0

    if total_buy_qty > 0:
        my_orders.setdefault(product, []).append(
            Order(product, int(bid_price), int(total_buy_qty))
        )
    if total_sell_qty > 0:
        my_orders.setdefault(product, []).append(
            Order(product, int(ask_price), int(-total_sell_qty))
        )


# ============================================================================
# STRATEGY BASE
# ============================================================================

class Strategy:
    name: str = "base"
    products: List[str] = []

    def init_state(self, state: TradingState, my_state: Dict[str, Any]):
        pass

    def run(self, state: TradingState, my_state: Dict[str, Any],
            my_orders: Dict[str, List[Order]]) -> None:
        raise NotImplementedError


# ============================================================================
# SNACKPACK SPREAD STRATEGY (CHOC-VAN, STRAW-RASP)
# ============================================================================

@dataclass
class SpreadConfig:
    leg_a: str
    leg_b: str
    prior_mean: float
    prior_std: float
    ema_alpha: float
    z_entry_min: float
    z_entry_full: float
    z_take_threshold: float


SPREAD_CONFIGS = [
    # CHOC - VAN: very stable, but corrected priors based on actual data
    # CHOC mean ≈ 9843, VAN mean ≈ 10097, so CHOC - VAN ≈ -254
    # σ(CHOC+VAN) was 76 (sum); σ(CHOC-VAN) ≈ 2*σ_individual / sqrt(corr stuff)
    # measured σ(CHOC-VAN) ≈ 65
    SpreadConfig(
        leg_a="SNACKPACK_CHOCOLATE",
        leg_b="SNACKPACK_VANILLA",
        prior_mean=-254.0,
        prior_std=65.0,
        ema_alpha=1.0 / 5000,
        z_entry_min=0.1,
        z_entry_full=0.7,
        z_take_threshold=3.3,
    ),
    # STRAW - RASP: less stable, drifts more.
    # STRAW mean ~10707, RASP mean ~10078, diff ≈ 629
    # Measured σ ≈ 200 (will track via EMA so this prior less critical)
    SpreadConfig(
        leg_a="SNACKPACK_STRAWBERRY",
        leg_b="SNACKPACK_RASPBERRY",
        prior_mean=629.0,
        prior_std=200.0,
        ema_alpha=1.0 / 3000,    # was 1/500 — track drift faster
        z_entry_min=0.5,
        z_entry_full=0.7,
        z_take_threshold=1.7,
    ),
]


class SnackpackSpreadStrategy(Strategy):
    name = "snackpacks_spreads_v4"
    products = [
        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
        "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
    ]

    def init_state(self, state: TradingState, my_state: Dict[str, Any]):
        my_state.setdefault("spreads", {})
        for cfg in SPREAD_CONFIGS:
            key = cfg.leg_a
            if key not in my_state["spreads"]:
                my_state["spreads"][key] = {
                    "ema_mean": cfg.prior_mean,
                    "ema_var": cfg.prior_std ** 2,
                    "initialized": False,
                    "n_ticks": 0,
                    "last_value": cfg.prior_mean,
                }

    def run(self, state: TradingState, my_state: Dict[str, Any],
            my_orders: Dict[str, List[Order]]) -> None:
        self.init_state(state, my_state)
        for cfg in SPREAD_CONFIGS:
            self._run_spread(state, my_state, my_orders, cfg)

    def _run_spread(self, state: TradingState, my_state: Dict[str, Any],
                    my_orders: Dict[str, List[Order]], cfg: SpreadConfig):
        if cfg.leg_a not in state.order_depths or cfg.leg_b not in state.order_depths:
            return
        depth_a = state.order_depths[cfg.leg_a]
        depth_b = state.order_depths[cfg.leg_b]
        bb_a, ba_a, mid_a = best_bid_ask_mid(depth_a)
        bb_b, ba_b, mid_b = best_bid_ask_mid(depth_b)
        if mid_a is None or mid_b is None:
            return

        spread_value = mid_a - mid_b
        s_state = my_state["spreads"][cfg.leg_a]
        update_ema(s_state, spread_value, cfg.ema_alpha)
        z = compute_z_blended(s_state, cfg.prior_mean, cfg.prior_std, cfg.ema_alpha)
        magnitude = z_to_magnitude(z, cfg.z_entry_min, cfg.z_entry_full)

        if z > 0:
            target_a = -magnitude
            target_b = +magnitude
        elif z < 0:
            target_a = +magnitude
            target_b = -magnitude
        else:
            target_a = 0
            target_b = 0

        cross_allowed = abs(z) >= cfg.z_take_threshold

        execute_leg_capped(cfg.leg_a, depth_a, bb_a, ba_a, target_a,
                           state.position.get(cfg.leg_a, 0),
                           cross_allowed, my_orders)
        execute_leg_capped(cfg.leg_b, depth_b, bb_b, ba_b, target_b,
                           state.position.get(cfg.leg_b, 0),
                           cross_allowed, my_orders)

        a_short = cfg.leg_a.replace("SNACKPACK_", "")[:3]
        b_short = cfg.leg_b.replace("SNACKPACK_", "")[:3]
        print(f"[snack:{a_short}-{b_short}] spr={spread_value:+.0f} "
              f"ema={s_state['ema_mean']:+.0f} z={z:+.2f} mag={magnitude} "
              f"tgt_a={target_a} tgt_b={target_b} "
              f"pos_a={state.position.get(cfg.leg_a, 0)} "
              f"pos_b={state.position.get(cfg.leg_b, 0)}")


# ============================================================================
# PISTACHIO STANDALONE STRATEGY
# ============================================================================
#
# PIS doesn't have a stable partner relationship across days, but it DOES
# mean-revert intraday (half-life ~850-1250 ticks). Daily mean drifts by
# about $100/day so we need a rolling EMA, not a fixed fair value.
#
# Same z-score sizing framework as the spreads, just on a single product.
# Daily std ≈ $130-145, so |z|=1 is about $130 from EMA mean.
# ============================================================================

@dataclass
class SoloConfig:
    product: str
    prior_mean: float
    prior_std: float
    ema_alpha: float
    z_entry_min: float
    z_entry_full: float
    z_take_threshold: float


SOLO_CONFIGS = [
    # PIS: range ~$600/day around a slowly-drifting mean
    # mean drifts: 9655 -> 9487 -> 9344 (about -$100/day)
    # daily std: 144, 142, 127 (avg ~138)
    # spread ~$16, so cross cost is meaningful
    SoloConfig(
        product="SNACKPACK_PISTACHIO",
        prior_mean=9500.0,    # halfway through observed drift range
        prior_std=140.0,
        ema_alpha=1.0 / 500,  # track drift but smooth out noise
        z_entry_min=0.1,
        z_entry_full=0.7,
        z_take_threshold=2.3,
    ),
]


class SolomeanReversionStrategy(Strategy):
    """Generic individual-product mean reversion. Anyone can be added."""
    name = "snackpacks_solo_v4"

    @property
    def products(self):
        return [c.product for c in SOLO_CONFIGS]

    def init_state(self, state: TradingState, my_state: Dict[str, Any]):
        my_state.setdefault("products", {})
        for cfg in SOLO_CONFIGS:
            if cfg.product not in my_state["products"]:
                my_state["products"][cfg.product] = {
                    "ema_mean": cfg.prior_mean,
                    "ema_var": cfg.prior_std ** 2,
                    "initialized": False,
                    "n_ticks": 0,
                    "last_value": cfg.prior_mean,
                }

    def run(self, state: TradingState, my_state: Dict[str, Any],
            my_orders: Dict[str, List[Order]]) -> None:
        self.init_state(state, my_state)
        for cfg in SOLO_CONFIGS:
            self._run_solo(state, my_state, my_orders, cfg)

    def _run_solo(self, state: TradingState, my_state: Dict[str, Any],
                  my_orders: Dict[str, List[Order]], cfg: SoloConfig):
        if cfg.product not in state.order_depths:
            return
        depth = state.order_depths[cfg.product]
        bb, ba, mid = best_bid_ask_mid(depth)
        if mid is None:
            return

        p_state = my_state["products"][cfg.product]
        update_ema(p_state, mid, cfg.ema_alpha)
        z = compute_z_blended(p_state, cfg.prior_mean, cfg.prior_std, cfg.ema_alpha)
        magnitude = z_to_magnitude(z, cfg.z_entry_min, cfg.z_entry_full)

        # z > 0: price > mean → expect reversion down → SHORT
        # z < 0: price < mean → expect reversion up → LONG
        if z > 0:
            target_pos = -magnitude
        elif z < 0:
            target_pos = +magnitude
        else:
            target_pos = 0

        cross_allowed = abs(z) >= cfg.z_take_threshold

        execute_leg_capped(cfg.product, depth, bb, ba, target_pos,
                           state.position.get(cfg.product, 0),
                           cross_allowed, my_orders)

        short_name = cfg.product.replace("SNACKPACK_", "")[:3]
        print(f"[solo:{short_name}] mid={mid:.0f} ema={p_state['ema_mean']:.0f} "
              f"z={z:+.2f} mag={magnitude} tgt={target_pos} "
              f"pos={state.position.get(cfg.product, 0)}")


# ============================================================================
# TRADER
# ============================================================================

class Trader:
    def __init__(self):
        self.strategies: List[Strategy] = [
            SnackpackSpreadStrategy(),
            SolomeanReversionStrategy(),
        ]

    def run(self, state: TradingState):
        saved = SavedState.load(state.traderData)
        my_orders: Dict[str, List[Order]] = {}

        for strat in self.strategies:
            strat_state = saved.get(strat.name)
            try:
                strat.run(state, strat_state, my_orders)
            except Exception as e:
                print(f"[ERR strategy={strat.name}] {e}")
            saved.put(strat.name, strat_state)

        # Aggregate cap per product (final defense)
        clean_orders = {}
        for product, orders in my_orders.items():
            current_pos = state.position.get(product, 0)
            buy_cap = max(0, POSITION_LIMIT - current_pos)
            sell_cap = max(0, POSITION_LIMIT + current_pos)

            buys = sorted([o for o in orders if o.quantity > 0], key=lambda o: o.price)
            sells = sorted([o for o in orders if o.quantity < 0], key=lambda o: -o.price)

            trimmed_buys = []
            remaining = buy_cap
            for o in buys:
                if remaining <= 0:
                    break
                q = min(o.quantity, remaining)
                if q > 0:
                    trimmed_buys.append(Order(product, o.price, int(q)))
                    remaining -= q

            trimmed_sells = []
            remaining = sell_cap
            for o in sells:
                if remaining <= 0:
                    break
                q = min(-o.quantity, remaining)
                if q > 0:
                    trimmed_sells.append(Order(product, o.price, int(-q)))
                    remaining -= q

            combined = trimmed_buys + trimmed_sells

            if trimmed_buys and trimmed_sells:
                min_sell_price = min(o.price for o in trimmed_sells)
                combined = [o for o in combined
                            if not (o.quantity > 0 and o.price >= min_sell_price)]

            clean_orders[product] = combined

        return clean_orders, 0, saved.dump()