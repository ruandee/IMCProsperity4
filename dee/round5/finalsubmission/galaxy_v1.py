"""
galaxy_v0.py — Galaxy Sounds basket trader

No basket/spread structure: returns correlations within Galaxy Sounds are
all ~0. Same situation as sleep pods — 5 independent products.

But unlike sleep pods, only 2 of these 5 are tradeable as mean reverters:

TIER 1 — Mean reversion + MM (long EMA, trust prior):
  - GALAXY_SOUNDS_DARK_MATTER   drift 0.34%, ADF max 0.50 (stable!)
  - GALAXY_SOUNDS_SOLAR_FLAMES  drift 1.56%, ADF min 0.25 (most stationary)

TIER 2 — MM-only (no directional signal, just collect spread):
  - GALAXY_SOUNDS_PLANETARY_RINGS  drift 6.6%, ADF p=0.98 d2 (random walk)
  - GALAXY_SOUNDS_SOLAR_WINDS      drift 8.6%, ADF max 0.72
  - GALAXY_SOUNDS_BLACK_HOLES      drift 18%, never stationary

Spreads $13-15 — wider than sleep pods. MM economics are good per fill
but fills will be rarer.
"""

from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import math


POSITION_LIMIT = 10


# ============================================================================
# State management
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
# Utilities
# ============================================================================

def best_bid_ask_mid(order_depth: OrderDepth):
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None, None, None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return best_bid, best_ask, (best_bid + best_ask) / 2.0


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
# EMA helpers
# ============================================================================

def update_ema(state: dict, value: float, alpha: float):
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
    std_emp = math.sqrt(max(state["ema_var"], 1.0))
    warmup_target = 3.0 / alpha
    blend = min(1.0, state["n_ticks"] / warmup_target)
    std = blend * std_emp + (1 - blend) * prior_std
    mean = blend * state["ema_mean"] + (1 - blend) * prior_mean
    if std <= 0:
        return 0.0
    return (state["last_value"] - mean) / std


def z_to_magnitude(z: float, z_min: float, z_full: float) -> int:
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
                       mm_base_size: int = 3,
                       mm_max_deviation: int = 2,
                       cross_min_gap: int = 5):
    """Mean reversion + MM-at-target. Used for Tier 1 products."""
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


def execute_mm_only(product: str, depth: OrderDepth,
                    best_bid: int, best_ask: int,
                    current_pos: int,
                    my_orders: Dict[str, List[Order]],
                    mm_base_size: int = 3,
                    inventory_skew_max: int = 5):
    """Pure MM with inventory skew — no directional signal.
    
    Quote bid+1/ask-1 (or join if spread is 1).
    Inventory skew: when long, lean toward selling more (smaller buy size,
    larger sell size); when short, lean toward buying more.
    
    This avoids directional bias from any signal — we simply collect spread
    on every wiggle and gently target zero inventory.
    """
    buy_cap = max(0, POSITION_LIMIT - current_pos)
    sell_cap = max(0, POSITION_LIMIT + current_pos)

    bid_price = best_bid + 1 if (best_ask - best_bid) > 1 else best_bid
    ask_price = best_ask - 1 if (best_ask - best_bid) > 1 else best_ask

    # Inventory skew: clamp to [-1, +1]
    # When current_pos > 0 (long), reduce buy size, keep sell size
    # When current_pos < 0 (short), keep buy size, reduce sell size
    pos_pct = max(-1.0, min(1.0, current_pos / inventory_skew_max))

    # Buy size shrinks as we get longer
    buy_size = int(mm_base_size * (1.0 - max(0, pos_pct)))
    # Sell size shrinks as we get shorter
    sell_size = int(mm_base_size * (1.0 + min(0, pos_pct)))

    buy_size = min(buy_size, buy_cap)
    sell_size = min(sell_size, sell_cap)

    if buy_size > 0:
        my_orders.setdefault(product, []).append(
            Order(product, int(bid_price), int(buy_size))
        )
    if sell_size > 0:
        my_orders.setdefault(product, []).append(
            Order(product, int(ask_price), int(-sell_size))
        )


# ============================================================================
# Strategy
# ============================================================================

@dataclass
class MeanRevConfig:
    """Tier 1: full mean reversion with z-score + MM-at-target."""
    product: str
    prior_mean: float
    prior_std: float
    ema_alpha: float
    z_entry_min: float
    z_entry_full: float
    z_take_threshold: float


@dataclass
class MMOnlyConfig:
    """Tier 2: pure passive MM, no directional signal."""
    product: str
    mm_base_size: int = 3
    inventory_skew_max: int = 5


# Tier 1: stable mean reverters. Long EMA, trust prior.
TIER_1_CONFIGS = [
    MeanRevConfig(
        product="GALAXY_SOUNDS_DARK_MATTER",
        prior_mean=10227.0,    # avg of d2/d3/d4 (drifts only 0.34%)
        prior_std=330.0,       # daily std ~330
        ema_alpha=1.0 / 1500,  # long EMA - prior is reliable
        z_entry_min=0.3,
        z_entry_full=1.0,
        z_take_threshold=2.0,
    ),
    MeanRevConfig(
        product="GALAXY_SOUNDS_SOLAR_FLAMES",
        prior_mean=11093.0,
        prior_std=450.0,
        ema_alpha=1.0 / 1500,
        z_entry_min=0.5,
        z_entry_full=1.5,
        z_take_threshold=2.5,
    ),
]

# Tier 2: pure MM. No directional signal — just collect spread.
TIER_2_CONFIGS = [
    # MMOnlyConfig(product="GALAXY_SOUNDS_PLANETARY_RINGS", mm_base_size=5, inventory_skew_max=7),
    # MMOnlyConfig(product="GALAXY_SOUNDS_SOLAR_WINDS", mm_base_size=4, inventory_skew_max=5),
    # MMOnlyConfig(product="GALAXY_SOUNDS_BLACK_HOLES", mm_base_size=5, inventory_skew_max=5),
]


class GalaxySoundsStrategy:
    name = "galaxy_v0"

    @property
    def products(self):
        return ([c.product for c in TIER_1_CONFIGS]
                + [c.product for c in TIER_2_CONFIGS])

    def init_state(self, my_state: Dict[str, Any]):
        my_state.setdefault("products", {})
        for cfg in TIER_1_CONFIGS:
            if cfg.product not in my_state["products"]:
                my_state["products"][cfg.product] = {
                    "ema_mean": cfg.prior_mean,
                    "ema_var": cfg.prior_std ** 2,
                    "initialized": False,
                    "n_ticks": 0,
                    "last_value": cfg.prior_mean,
                }
        # Tier 2 needs no state (stateless MM)

    def run(self, state: TradingState, my_state: Dict[str, Any],
            my_orders: Dict[str, List[Order]]) -> None:
        self.init_state(my_state)
        for cfg in TIER_1_CONFIGS:
            self._run_meanrev(state, my_state, my_orders, cfg)
        for cfg in TIER_2_CONFIGS:
            self._run_mm_only(state, my_orders, cfg)

    def _run_meanrev(self, state: TradingState, my_state: Dict[str, Any],
                     my_orders: Dict[str, List[Order]], cfg: MeanRevConfig):
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

        short = "DAR" if "DARK" in cfg.product else "SOL_F"
        print(f"[gs:T1:{short}] mid={mid:.0f} ema={p_state['ema_mean']:.0f} "
              f"z={z:+.2f} mag={magnitude} tgt={target_pos} "
              f"pos={state.position.get(cfg.product, 0)}")

    def _run_mm_only(self, state: TradingState,
                     my_orders: Dict[str, List[Order]], cfg: MMOnlyConfig):
        if cfg.product not in state.order_depths:
            return
        depth = state.order_depths[cfg.product]
        bb, ba, mid = best_bid_ask_mid(depth)
        if mid is None:
            return

        execute_mm_only(cfg.product, depth, bb, ba,
                        state.position.get(cfg.product, 0),
                        my_orders, cfg.mm_base_size, cfg.inventory_skew_max)

        if "PLANETARY" in cfg.product:
            short = "PLA"
        elif "WINDS" in cfg.product:
            short = "SOL_W"
        else:
            short = "BLA"
        print(f"[gs:T2:{short}] mid={mid:.0f} pos={state.position.get(cfg.product, 0)} "
              f"bid={bb} ask={ba}")


# ============================================================================
# Trader
# ============================================================================

class Trader:
    def __init__(self):
        self.strategies = [GalaxySoundsStrategy()]

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

        # Per-product cap aggregation
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