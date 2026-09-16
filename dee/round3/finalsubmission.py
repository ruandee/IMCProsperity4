from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
import math
import json

# =============================================================================
# ── OPTIONS STRATEGY CONSTANTS  (submission.py) ──────────────────────────────
# =============================================================================
TICKS_PER_DAY  = 999_000
TTE_START_DAYS = 5.0
UNDERLYING     = "VELVETFRUIT_EXTRACT"

SIGNAL_STRIKES      = [5100, 5200, 5300, 5400]
VANNA_HEDGE_STRIKES = [5000, 5500]
ALL_STRIKES         = SIGNAL_STRIKES + VANNA_HEDGE_STRIKES
SYMS                = {K: f"VEV_{K}" for K in ALL_STRIKES}

VOUCHER_LIMIT  = 300
VFE_LIMIT      = 200

DELTA_TARGET   = 300
DELTA_TOL      = 10

SMILE_A = 0.05128
SMILE_B = -0.103242
SMILE_C = 0.063

TIGHT_SPREAD_MAX     = 2
DECAY_FACTOR         = 0.985

SCALP_BASE           = 300
EDGE_INSIDE          = 1

IV_NOISE_FLOOR       = 0.0005
TIGHT_MULT           = 0.5
LOOSE_MULT           = 2.0

VANNA_TOL              = 10
VANNA_HEDGE_PASSIVE    = 40
VANNA_HEDGE_AGGRESSIVE = 300

SPIKE_RESIDUAL_IV      = 0.012
SPIKE_CHANGE_IV        = 0.005

LADDER_LEVELS          = 3
LADDER_FRACS           = [0.5, 0.3, 0.2]

# =============================================================================
# ── VELVET SPOT STRATEGY CONSTANTS  (message.txt) ────────────────────────────
# =============================================================================
VEL_POSITION_LIMIT = 300
VEL_FAIR_WINDOW    = 200
VEL_MIN_HISTORY    = 80
VEL_TAKE_EDGE      = 8.0
VEL_CLIP           = 5
VEL_MAX_STORED_MIDS = 250
VEL_INVENTORY_SKEW  = 6.0

# =============================================================================
# ── HYDROGEL STRATEGY CONSTANTS  (gelpackfinal.py) ───────────────────────────
# =============================================================================
HYDROGEL_PRODUCT = "HYDROGEL_PACK"

LONG_MEAN = 9991.0
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

GAP_TAKE_THRESHOLD = 150
GAP_NEUTRAL_BAND   = 30

DRAWDOWN_THRESHOLD      = 1500
LOCKDOWN_DURATION       = 500
LOCKDOWN_POS_CAP        = 30

HG_POSITION_LIMIT = 200
RISK_FACTOR        = 0.05
MAKING_EDGE        = 1

# =============================================================================
# ── BLACK-SCHOLES HELPERS ─────────────────────────────────────────────────────
# =============================================================================
_SQRT2   = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

def _npdf(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))

def bs_price(S, K, T_days, s_daily, is_call=True):
    if S <= 0 or K <= 0: return 0.0
    if T_days < 1e-9 or s_daily < 1e-9:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sq = s_daily * math.sqrt(T_days)
    d1 = (math.log(S / K) + 0.5 * s_daily * s_daily * T_days) / sq
    d2 = d1 - sq
    if is_call:
        return S * _ncdf(d1) - K * _ncdf(d2)
    else:
        return K * _ncdf(-d2) - S * _ncdf(-d1)

def bs_delta(S, K, T_days, s_daily, is_call=True):
    if S <= 0 or K <= 0: return 0.0
    if T_days < 1e-9 or s_daily < 1e-9:
        if is_call:
            return 1.0 if S >= K else 0.0
        else:
            return -1.0 if S <= K else 0.0
    d1 = (math.log(S / K) + 0.5 * s_daily * s_daily * T_days) / (s_daily * math.sqrt(T_days))
    return _ncdf(d1) if is_call else _ncdf(d1) - 1.0

def bs_vega(S, K, T_days, s_daily):
    if S <= 0 or K <= 0: return 0.0
    if T_days < 1e-9 or s_daily < 1e-9: return 0.0
    sq = s_daily * math.sqrt(T_days)
    d1 = (math.log(S / K) + 0.5 * s_daily * s_daily * T_days) / sq
    return S * _npdf(d1) * math.sqrt(T_days)

def bs_vanna(S, K, T_days, s_daily):
    if S <= 0 or K <= 0: return 0.0
    if T_days < 1e-9 or s_daily < 1e-9: return 0.0
    sq = s_daily * math.sqrt(T_days)
    d1 = (math.log(S / K) + 0.5 * s_daily * s_daily * T_days) / sq
    return bs_vega(S, K, T_days, s_daily) * (1.0 - d1 / sq) / S

def implied_vol(Price, S, K, T_days, is_call=True, s0=0.015):
    if S <= 0 or K <= 0: return None
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if Price <= intrinsic + 1e-5: return None
    s = s0
    for _ in range(30):
        p = bs_price(S, K, T_days, s, is_call)
        v = bs_vega(S, K, T_days, s)
        e = p - Price
        if abs(e) < 1e-4: return s
        if v < 1e-10: break
        s = max(0.0005, min(0.3, s - e / v))
    lo, hi = 0.0005, 0.3
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if bs_price(S, K, T_days, m, is_call) < Price:
            lo = m
        else:
            hi = m
        if hi - lo < 1e-6:
            return m
    return s

def _best_bid(od):
    if not od.buy_orders: return None, 0
    p = max(od.buy_orders); return p, od.buy_orders[p]

def _best_ask(od):
    if not od.sell_orders: return None, 0
    p = min(od.sell_orders); return p, od.sell_orders[p]

# =============================================================================
# ── HYDROGEL STATE DATACLASS ──────────────────────────────────────────────────
# =============================================================================
@dataclass
class HydrogelState:
    price_history: list = field(default_factory=list)
    last_mid: float | None = None
    last_pos: int = 0
    cum_pnl: float = 0.0
    peak_pnl: float = 0.0
    in_lockdown: bool = False
    lockdown_ticks_remaining: int = 0

# =============================================================================
# ── COMBINED TRADER ───────────────────────────────────────────────────────────
# =============================================================================
class Trader:

    def run(self, state: TradingState):
        # ── Load combined state ──────────────────────────────────────────────
        try:
            raw = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            raw = {}

        options_ps   = raw.get("options", {})
        velvet_mem   = raw.get("velvet", {})
        hydrogel_raw = raw.get("hydrogel", {})

        # ── Run all three sub-strategies ─────────────────────────────────────
        result: Dict[str, List[Order]] = {}

        options_orders, options_ps_out = self._run_options(state, options_ps)
        for sym, orders in options_orders.items():
            result.setdefault(sym, []).extend(orders)

        velvet_orders, velvet_mem_out = self._run_velvet(state, velvet_mem)
        if velvet_orders:
            result.setdefault(UNDERLYING, []).extend(velvet_orders)

        hydrogel_orders, hydrogel_state_out = self._run_hydrogel(state, hydrogel_raw)
        if hydrogel_orders:
            result[HYDROGEL_PRODUCT] = hydrogel_orders

        # ── Persist combined state ───────────────────────────────────────────
        trader_data = json.dumps(
            {"options": options_ps_out, "velvet": velvet_mem_out, "hydrogel": hydrogel_state_out},
            separators=(",", ":")
        )

        return result, 0, trader_data

    # =========================================================================
    # OPTIONS STRATEGY  (submission.py)
    # =========================================================================
    def _run_options(self, state: TradingState, ps: dict) -> Tuple[Dict[str, List[Order]], dict]:
        result: Dict[str, List[Order]] = {sym: [] for sym in SYMS.values()}
        result[UNDERLYING] = []

        iv_seeds      = ps.get("ivs", {str(K): 0.24 for K in ALL_STRIKES})
        t_decay       = float(ps.get("t", 1.0))
        prev_residuals = ps.get("pr", {})

        tick   = int(state.timestamp)
        T_days = max((TTE_START_DAYS - tick / TICKS_PER_DAY), 1e-9)
        sqrt_T = math.sqrt(T_days)

        S_obs  = None
        vfe_bp: Optional[int] = None
        vfe_ap: Optional[int] = None
        if UNDERLYING in state.order_depths:
            od_v = state.order_depths[UNDERLYING]
            bp_v, _ = _best_bid(od_v)
            ap_v, _ = _best_ask(od_v)
            if bp_v is not None and ap_v is not None:
                S_obs  = 0.5 * (bp_v + ap_v)
                vfe_bp = bp_v
                vfe_ap = ap_v
        S_eff = S_obs if S_obs is not None else 5250.0

        live_bids: Dict[int, Tuple[int, int]] = {}
        live_asks: Dict[int, Tuple[int, int]] = {}
        live_mids: Dict[int, float] = {}
        live_iv_obs:  Dict[int, float] = {}
        live_iv_adj:  Dict[int, float] = {}

        for K in ALL_STRIKES:
            sym = SYMS[K]
            if sym not in state.order_depths: continue
            od = state.order_depths[sym]
            bp, bq = _best_bid(od)
            ap, aq = _best_ask(od)
            if bp is None or ap is None: continue
            live_bids[K] = (bp, bq)
            live_asks[K] = (ap, aq)
            live_mids[K] = 0.5 * (bp + ap)
            seed = float(iv_seeds.get(str(K), 0.24))
            iv = implied_vol(live_mids[K], S_eff, K, T_days, is_call=True, s0=seed)
            if iv is not None:
                iv_seeds[str(K)] = iv
                live_iv_obs[K] = iv
                live_iv_adj[K] = iv

        sp_5200 = (live_asks[5200][0] - live_bids[5200][0]) if 5200 in live_bids and 5200 in live_asks else 999
        sp_5300 = (live_asks[5300][0] - live_bids[5300][0]) if 5300 in live_bids and 5300 in live_asks else 999
        tight_surface = (sp_5200 <= TIGHT_SPREAD_MAX) and (sp_5300 <= TIGHT_SPREAD_MAX)

        if tight_surface:
            t_decay = 1.0
        else:
            t_decay = max(0.05, t_decay * DECAY_FACTOR)

        regime_mult = TIGHT_MULT if tight_surface else LOOSE_MULT

        residuals: Dict[int, float] = {}
        vegas:     Dict[int, float] = {}
        deltas:    Dict[int, float] = {}
        vannas:    Dict[int, float] = {}
        thresh:    Dict[int, float] = {}
        sides:     Dict[int, int]   = {}

        max_vega = 1.0
        for K in SIGNAL_STRIKES:
            if K not in live_iv_adj: continue
            v = bs_vega(S_eff, K, T_days, live_iv_adj[K])
            if v > max_vega: max_vega = v

        for K in ALL_STRIKES:
            if K not in live_iv_adj: continue
            iv_a = live_iv_adj[K]
            iv_o = live_iv_obs[K]
            m       = math.log(K / S_eff) / sqrt_T
            iv_fit  = SMILE_A * m * m + SMILE_B * m + SMILE_C
            r       = iv_a - iv_fit
            residuals[K] = r
            vegas[K]  = bs_vega(S_eff, K, T_days, iv_o)
            deltas[K] = bs_delta(S_eff, K, T_days, iv_o, is_call=True)
            vannas[K] = bs_vanna(S_eff, K, T_days, iv_o)
            half_sp = 0.0
            if K in live_bids and K in live_asks:
                half_sp = (live_asks[K][0] - live_bids[K][0]) / 2.0
            cost_thresh = half_sp / vegas[K] if vegas[K] > 1e-6 else 1e6
            thresh[K]   = max(cost_thresh, regime_mult * IV_NOISE_FLOOR)
            if K in SIGNAL_STRIKES and abs(r) > thresh[K]:
                sides[K] = -1 if r > 0 else +1
            else:
                sides[K] = 0

        targets: Dict[int, int] = {K: 0 for K in ALL_STRIKES}
        for K in SIGNAL_STRIKES:
            if sides.get(K, 0) == 0: continue
            v = vegas.get(K, 0.0)
            atm_weight = v / max_vega if max_vega > 0 else 0.0
            size = int(round(SCALP_BASE * t_decay * atm_weight))
            size = max(0, min(VOUCHER_LIMIT, size))
            targets[K] = sides[K] * size

        net_vanna = sum(targets[K] * vannas.get(K, 0.0) for K in ALL_STRIKES)
        if abs(net_vanna) > VANNA_TOL:
            best_K, best_cost = None, float('inf')
            for K in SIGNAL_STRIKES:
                if targets[K] == 0: continue
                contrib = targets[K] * vannas.get(K, 0.0)
                if (contrib > 0 and net_vanna > 0) or (contrib < 0 and net_vanna < 0):
                    cost_per_unit = abs(residuals.get(K, 0.0)) * vegas.get(K, 0.0)
                    if cost_per_unit < best_cost:
                        best_cost = cost_per_unit
                        best_K    = K
            if best_K is not None:
                vanna_K = vannas.get(best_K, 0.0)
                if abs(vanna_K) > 1e-9:
                    qty_remove = net_vanna / vanna_K
                    new_t = targets[best_K] - int(round(qty_remove))
                    if (targets[best_K] > 0 and new_t < 0) or (targets[best_K] < 0 and new_t > 0):
                        new_t = 0
                    new_t = max(-VOUCHER_LIMIT, min(VOUCHER_LIMIT, new_t))
                    targets[best_K] = new_t

        positions = state.position
        vfe_pos   = positions.get(UNDERLYING, 0)

        proj_net_vanna = (
            sum(targets[K] * vannas.get(K, 0.0) for K in SIGNAL_STRIKES)
            + sum(positions.get(SYMS[K], 0) * vannas.get(K, 0.0) for K in VANNA_HEDGE_STRIKES)
        )

        prev_r5200 = float(prev_residuals.get("5200", 0.0))
        prev_r5300 = float(prev_residuals.get("5300", 0.0))
        cur_r5200  = residuals.get(5200, 0.0)
        cur_r5300  = residuals.get(5300, 0.0)
        spike_detected = (
            abs(cur_r5200) > SPIKE_RESIDUAL_IV or
            abs(cur_r5300) > SPIKE_RESIDUAL_IV or
            abs(cur_r5200 - prev_r5200) > SPIKE_CHANGE_IV or
            abs(cur_r5300 - prev_r5300) > SPIKE_CHANGE_IV
        )

        if abs(proj_net_vanna) > VANNA_TOL:
            ranked_hedge = sorted(VANNA_HEDGE_STRIKES, key=lambda k: -abs(vannas.get(k, 0.0)))
            for K in ranked_hedge:
                if abs(proj_net_vanna) <= VANNA_TOL: break
                if K not in live_bids or K not in live_asks: continue
                vanna_K = vannas.get(K, 0.0)
                if abs(vanna_K) < 1e-6: continue
                pos_K = positions.get(SYMS[K], 0)
                bp, bq = live_bids[K]
                ap, aq = live_asks[K]
                if ap - bp < 2: continue
                if proj_net_vanna > 0:
                    direction = -1 if vanna_K > 0 else +1
                else:
                    direction = +1 if vanna_K > 0 else -1
                qty_full = int(round(abs(proj_net_vanna / vanna_K)))
                room = (VOUCHER_LIMIT - pos_K) if direction > 0 else (VOUCHER_LIMIT + pos_K)
                if room <= 0: continue
                if spike_detected:
                    qty = min(qty_full, room, VANNA_HEDGE_AGGRESSIVE,
                              (-aq if direction > 0 and aq < 0 else
                               bq  if direction < 0 and bq > 0 else
                               VANNA_HEDGE_AGGRESSIVE))
                    if qty <= 0: continue
                    if direction > 0:
                        result[SYMS[K]].append(Order(SYMS[K], ap, qty))
                    else:
                        result[SYMS[K]].append(Order(SYMS[K], bp, -qty))
                    proj_net_vanna += direction * qty * vanna_K
                else:
                    qty = min(qty_full, room, VANNA_HEDGE_PASSIVE)
                    if qty <= 0: continue
                    if direction > 0:
                        price = bp + EDGE_INSIDE
                        if price < ap:
                            result[SYMS[K]].append(Order(SYMS[K], price, qty))
                            proj_net_vanna += qty * vanna_K
                    else:
                        price = ap - EDGE_INSIDE
                        if price > bp:
                            result[SYMS[K]].append(Order(SYMS[K], price, -qty))
                            proj_net_vanna -= qty * vanna_K

        buy_qty:  Dict[str, int] = {sym: 0 for sym in result}
        sell_qty: Dict[str, int] = {sym: 0 for sym in result}

        for K in SIGNAL_STRIKES:
            sym = SYMS[K]
            pos = positions.get(sym, 0)
            tgt = targets[K]
            gap = tgt - pos
            if gap == 0: continue
            if K not in live_bids or K not in live_asks: continue
            bp, bq = live_bids[K]
            ap, aq = live_asks[K]
            if ap - bp < 2: continue

            r_K = residuals.get(K, 0.0)
            th  = thresh.get(K, 0.0)

            if gap > 0:
                buy_room = VOUCHER_LIMIT - pos - buy_qty[sym]
                if buy_room <= 0: continue
                remaining = min(gap, buy_room)
                if r_K < -th and aq < 0:
                    take = min(remaining, -aq)
                    if take > 0:
                        result[sym].append(Order(sym, ap, take))
                        buy_qty[sym] += take
                        remaining -= take
                if remaining > 0:
                    max_inside = ap - 1
                    issued = 0
                    for level_idx in range(LADDER_LEVELS):
                        if remaining - issued <= 0: break
                        price = bp + EDGE_INSIDE + level_idx
                        if price > max_inside: break
                        frac = LADDER_FRACS[level_idx] if level_idx < len(LADDER_FRACS) else 0.0
                        level_qty = int(round(remaining * frac))
                        if level_idx == LADDER_LEVELS - 1:
                            level_qty = remaining - issued
                        level_qty = min(level_qty, remaining - issued)
                        if level_qty > 0:
                            result[sym].append(Order(sym, price, level_qty))
                            buy_qty[sym] += level_qty
                            issued += level_qty
            else:
                sell_room = VOUCHER_LIMIT + pos - sell_qty[sym]
                if sell_room <= 0: continue
                remaining = min(-gap, sell_room)
                if r_K > th and bq > 0:
                    take = min(remaining, bq)
                    if take > 0:
                        result[sym].append(Order(sym, bp, -take))
                        sell_qty[sym] += take
                        remaining -= take
                if remaining > 0:
                    min_inside = bp + 1
                    issued = 0
                    for level_idx in range(LADDER_LEVELS):
                        if remaining - issued <= 0: break
                        price = ap - EDGE_INSIDE - level_idx
                        if price < min_inside: break
                        frac = LADDER_FRACS[level_idx] if level_idx < len(LADDER_FRACS) else 0.0
                        level_qty = int(round(remaining * frac))
                        if level_idx == LADDER_LEVELS - 1:
                            level_qty = remaining - issued
                        level_qty = min(level_qty, remaining - issued)
                        if level_qty > 0:
                            result[sym].append(Order(sym, price, -level_qty))
                            sell_qty[sym] += level_qty
                            issued += level_qty

        # # Delta hedge on VELVETFRUIT_EXTRACT
        # net_delta_opts = sum(
        #     positions.get(SYMS[K], 0) * deltas.get(K, 0.0) for K in ALL_STRIKES
        # )
        # net_delta = net_delta_opts + vfe_pos
        # delta_gap = DELTA_TARGET - net_delta

        # if abs(delta_gap) > DELTA_TOL and vfe_bp is not None and vfe_ap is not None:
        #     qty_raw = int(round(abs(delta_gap)))
        #     if delta_gap > 0:
        #         buy_room = VFE_LIMIT - vfe_pos
        #         qty = max(0, min(qty_raw, buy_room))
        #         if qty > 0:
        #             price = vfe_bp + EDGE_INSIDE
        #             if price < vfe_ap:
        #                 result[UNDERLYING].append(Order(UNDERLYING, price, qty))
        #     else:
        #         sell_room = VFE_LIMIT + vfe_pos
        #         qty = max(0, min(qty_raw, sell_room))
        #         if qty > 0:
        #             price = vfe_ap - EDGE_INSIDE
        #             if price > vfe_bp:
        #                 result[UNDERLYING].append(Order(UNDERLYING, price, -qty))

        # Position-limit safety
        for K in ALL_STRIKES:
            sym = SYMS[K]
            pos = positions.get(sym, 0)
            tot_b = sum(o.quantity  for o in result[sym] if o.quantity > 0)
            tot_s = sum(-o.quantity for o in result[sym] if o.quantity < 0)
            if pos + tot_b > VOUCHER_LIMIT or pos - tot_s < -VOUCHER_LIMIT:
                result[sym] = []

        u_b = sum(o.quantity  for o in result[UNDERLYING] if o.quantity > 0)
        u_s = sum(-o.quantity for o in result[UNDERLYING] if o.quantity < 0)
        if vfe_pos + u_b > VFE_LIMIT or vfe_pos - u_s < -VFE_LIMIT:
            result[UNDERLYING] = []

        # Logging
        # if tick % 20000 == 0:
        #     net_d  = net_delta
        #     net_v  = sum(positions.get(SYMS[K], 0) * vegas.get(K, 0.0)  for K in ALL_STRIKES)
        #     net_va = sum(positions.get(SYMS[K], 0) * vannas.get(K, 0.0) for K in ALL_STRIKES)
        #     regime = "TIGHT" if tight_surface else "loose"
        #     pos_str = " ".join(f"{K}:{positions.get(SYMS[K],0):+d}/{targets[K]:+d}" for K in ALL_STRIKES)
        #     pos_str += f" VFE:{vfe_pos:+d}"
        #     res_str = " ".join(f"{K}:{residuals.get(K,0)*1000:+.1f}m" for K in SIGNAL_STRIKES)
        #     n_orders = sum(len(result[s]) for s in result)
        #     print(
        #         f"t={tick} T_days={T_days:.4f} S={S_eff:.0f} reg={regime} t_dec={t_decay:.2f} "
        #         f"sp52={sp_5200} sp53={sp_5300} "
        #         f"netΔ={net_d:.0f} netV={net_v:.0f} netVa={net_va:.1f} ord={n_orders} "
        #         f"res=[{res_str}] pos/tgt=[{pos_str}]"
        #     )

        ps_out = {
            "ivs": iv_seeds,
            "t":   t_decay,
            "pr":  {str(K): residuals.get(K, 0.0) for K in [5200, 5300]},
        }
        return result, ps_out

    # =========================================================================
    # VELVET SPOT STRATEGY  (message.txt)
    # =========================================================================
    def _run_velvet(self, state: TradingState, memory: dict) -> Tuple[List[Order], dict]:
        depth    = state.order_depths.get(UNDERLYING)
        position = state.position.get(UNDERLYING, 0)
        ts       = state.timestamp

        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return [], memory

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        if best_bid >= best_ask:
            return [], memory

        mid = (best_bid + best_ask) / 2.0
        mids = memory.setdefault("velvet_mids", [])
        mids.append(mid)
        if len(mids) > VEL_MAX_STORED_MIDS:
            del mids[: len(mids) - VEL_MAX_STORED_MIDS]

        if len(mids) < VEL_MIN_HISTORY:
            return [], memory

        fair = sum(mids[-VEL_FAIR_WINDOW:]) / len(mids[-VEL_FAIR_WINDOW:])
        inventory_skew = (position / VEL_POSITION_LIMIT) * VEL_INVENTORY_SKEW
        skewed_fair    = fair - inventory_skew

        buy_capacity  = max(0, VEL_POSITION_LIMIT - position)
        sell_capacity = max(0, VEL_POSITION_LIMIT + position)

        buy_thr  = skewed_fair - VEL_TAKE_EDGE
        sell_thr = skewed_fair + VEL_TAKE_EDGE

        orders: List[Order] = []
        if best_ask < buy_thr and buy_capacity > 0:
            qty = min(VEL_CLIP, -depth.sell_orders[best_ask], buy_capacity)
            if qty > 0:
                orders.append(Order(UNDERLYING, best_ask, qty))
                #print(f"[VELVET ts={ts}] TAKE BUY {qty}@{best_ask} (fair={fair:.2f})")

        if best_bid > sell_thr and sell_capacity > 0:
            qty = min(VEL_CLIP, depth.buy_orders[best_bid], sell_capacity)
            if qty > 0:
                orders.append(Order(UNDERLYING, best_bid, -qty))
                #print(f"[VELVET ts={ts}] TAKE SELL {qty}@{best_bid} (fair={fair:.2f})")

        return orders, memory

    # =========================================================================
    # HYDROGEL STRATEGY  (gelpackfinal.py)
    # =========================================================================
    def _compute_trend_strength(self, mid_price, history):
        if len(history) < TREND_LOOKBACK:
            return 0.0
        recent = history[-TREND_LOOKBACK:]
        recent_change = mid_price - recent[0]
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        if not diffs:
            return 0.0
        variance = sum(d * d for d in diffs) / len(diffs)
        tick_std = variance ** 0.5
        expected_std = tick_std * (TREND_LOOKBACK ** 0.5) + 1e-9
        return recent_change / expected_std

    def _compute_target_position(self, mid_price, history):
        z_score  = (mid_price - LONG_MEAN) / LONG_STD
        momentum = mid_price - history[-SHORT_LOOKBACK] if len(history) >= SHORT_LOOKBACK else 0

        trend_strength = self._compute_trend_strength(mid_price, history)
        abs_trend = abs(trend_strength)

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
                dampening = max(0.0, 1.0 - (abs_trend - TREND_DAMPEN_THRESHOLD) /
                                             (TREND_FLIP_THRESHOLD - TREND_DAMPEN_THRESHOLD))
                target = int(baseline_target * dampening)
                regime += " | DAMPENED"
        else:
            target = baseline_target

        return target, regime, z_score, momentum, trend_strength

    def _run_hydrogel(self, state: TradingState, hydrogel_raw: dict) -> Tuple[List[Order], dict]:
        product = HYDROGEL_PRODUCT
        ps = HydrogelState(**hydrogel_raw) if hydrogel_raw else HydrogelState()
        orders: List[Order] = []

        if product not in state.order_depths:
            return orders, asdict(ps)

        order_depth = state.order_depths[product]
        pos = state.position.get(product, 0)
        original_pos = pos

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, asdict(ps)

        best_bid  = max(order_depth.buy_orders.keys())
        best_ask  = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0

        # PnL tracking
        if ps.last_mid is not None:
            mtm_change = ps.last_pos * (mid_price - ps.last_mid)
            ps.cum_pnl += mtm_change
            ps.peak_pnl = max(ps.peak_pnl, ps.cum_pnl)
        ps.last_mid = mid_price
        ps.last_pos = pos

        # Circuit breaker
        drawdown = ps.peak_pnl - ps.cum_pnl
        if not ps.in_lockdown and drawdown > DRAWDOWN_THRESHOLD:
            ps.in_lockdown = True
            ps.lockdown_ticks_remaining = LOCKDOWN_DURATION
            #print(f"!!! CIRCUIT BREAKER: drawdown={drawdown:.0f} from peak={ps.peak_pnl:.0f} -> LOCKDOWN")

        if ps.in_lockdown:
            ps.lockdown_ticks_remaining -= 1
            if ps.lockdown_ticks_remaining <= 0:
                ps.in_lockdown = False
                ps.peak_pnl = ps.cum_pnl
                #print("!!! LOCKDOWN ENDED, resetting peak")

        ps.price_history.append(mid_price)
        if len(ps.price_history) > MAX_HISTORY:
            ps.price_history = ps.price_history[-MAX_HISTORY:]

        target_pos, regime, z_score, momentum, trend = self._compute_target_position(
            mid_price, ps.price_history
        )

        if ps.in_lockdown:
            target_pos = max(-LOCKDOWN_POS_CAP, min(LOCKDOWN_POS_CAP, target_pos))
            regime += " | LOCKED"

        gap      = target_pos - pos
        buy_cap  = HG_POSITION_LIMIT - pos
        sell_cap = -HG_POSITION_LIMIT - pos

        # Aggressive take
        took_volume = 0
        if abs(gap) > GAP_TAKE_THRESHOLD:
            if gap > 0 and buy_cap > 0:
                remaining = min(gap, buy_cap)
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if remaining <= 0: break
                    fill = min(abs(order_depth.sell_orders[ask_price]), remaining)
                    if fill > 0:
                        orders.append(Order(product, ask_price, int(fill)))
                        pos      += fill
                        buy_cap  -= fill
                        remaining -= fill
                        took_volume += fill
            elif gap < 0 and sell_cap < 0:
                remaining = min(abs(gap), abs(sell_cap))
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if remaining <= 0: break
                    fill = min(abs(order_depth.buy_orders[bid_price]), remaining)
                    if fill > 0:
                        orders.append(Order(product, bid_price, int(-fill)))
                        pos      -= fill
                        sell_cap += fill
                        remaining -= fill
                        took_volume += fill

        # Passive make
        gap_after  = target_pos - pos
        res_price  = mid_price - ((pos - target_pos) * RISK_FACTOR)
        post_bid   = abs(gap_after) <= GAP_NEUTRAL_BAND or gap_after > 0
        post_ask   = abs(gap_after) <= GAP_NEUTRAL_BAND or gap_after < 0

        if post_bid and buy_cap > 0:
            bid_price = min(best_bid + 1, math.floor(res_price - MAKING_EDGE))
            orders.append(Order(product, int(bid_price), int(buy_cap)))

        if post_ask and sell_cap < 0:
            ask_price = max(best_ask - 1, math.ceil(res_price + MAKING_EDGE))
            orders.append(Order(product, int(ask_price), int(sell_cap)))

        # print(
        #     f"[HG] mid={mid_price:.1f} z={z_score:.2f} mom={momentum:.1f} trend={trend:.2f} "
        #     f"regime={regime} target={target_pos} pos={original_pos} took={took_volume} "
        #     f"cum_pnl={ps.cum_pnl:.0f} peak={ps.peak_pnl:.0f} dd={drawdown:.0f}"
        # )

        # Self-cross safety
        buy_list  = [o for o in orders if o.quantity > 0]
        sell_list = [o for o in orders if o.quantity < 0]
        if buy_list and sell_list:
            min_sell = min(o.price for o in sell_list)
            orders = [o for o in orders if not (o.quantity > 0 and o.price >= min_sell)]

        return orders, asdict(ps)