from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple, Optional
import math
import json

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TICKS_PER_DAY  = 999_000
TTE_START_DAYS = 5.0
UNDERLYING     = "VELVETFRUIT_EXTRACT"

# Strikes used for IV-deviation scalping (active alpha)
SIGNAL_STRIKES = [5100, 5200, 5300, 5400]

# Strikes used purely as vanna-neutralizers (passive sleeve)
VANNA_HEDGE_STRIKES = [5000, 5500]

ALL_STRIKES    = SIGNAL_STRIKES + VANNA_HEDGE_STRIKES
SYMS           = {K: f"VEV_{K}" for K in ALL_STRIKES}

VOUCHER_LIMIT  = 300
VFE_LIMIT      = 200

# ── Delta hedge on underlying ────────────────────────────────────────────
DELTA_TARGET   = 300          # net delta we want to hold
DELTA_TOL      = 10           # don't issue orders if gap is within this band

# ── Hard-coded IV smile (offline fit on 3M ticks) ────────────────────────
# (0.570140 * M^2) + (-1.153180 * M) + 0.742076
# new (0.690141 * M^2) + (-1.395898 * M) + 0.898267
# new new (0.051280 * M^2) + (-0.103242 * M) + 0.064080
# Fixed ? (0.086773 * M^2) + (-0.172874 * M) + 0.098169
SMILE_A = 0.086773
SMILE_B = -0.172874
SMILE_C = 0.098169

# ── Lo-Wang multiplicative adjustment ────────────────────────────────────
#LO_WANG_FACTOR = 1.0          # σ_adj = σ_observed · 0.82612

# ── Regime filter ────────────────────────────────────────────────────────
TIGHT_SPREAD_MAX = 2              # spread ≤ 2 on both 5200 and 5300
DECAY_FACTOR     = 0.95           # per-tick decay of signal strength

# ── Sizing ───────────────────────────────────────────────────────────────
SCALP_BASE       = 50             # base contracts per scalp at full t and ATM
EDGE_INSIDE      = 1              # ticks inside best bid/ask for passive

# ── Thresholds ───────────────────────────────────────────────────────────
IV_NOISE_FLOOR   = 0.002          # 10 bps IV minimum residual to consider
TIGHT_MULT       = 0.5            # tight regime: lower threshold
LOOSE_MULT       = 2.0            # loose regime: higher threshold

# ── Vanna neutrality tolerance & hedge sleeve params ────────────────────
VANNA_TOL              = 300         # neutralize when |net_vanna| > this
VANNA_HEDGE_PASSIVE    = 0         # contracts/tick passive hedge orders
VANNA_HEDGE_AGGRESSIVE = 0         # contracts/tick aggressive (cross) hedge

# ── IV-spike trigger on 5200/5300 (forces aggressive hedging) ───────────
SPIKE_RESIDUAL_IV      = 0.012      # |r_5200| or |r_5300| above this → spike
SPIKE_CHANGE_IV        = 0.005      # |Δr_5200| or |Δr_5300| above this → spike

# ── Order laddering (split target gap across price levels) ──────────────
LADDER_LEVELS          = 3          # 3 price levels per side
LADDER_FRACS           = [0.5, 0.3, 0.2]   # fraction of gap at each level

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────
_SQRT2   = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

def _npdf(x): 
    return math.exp(-0.5*x*x) / _SQRT2PI

def _ncdf(x): 
    return 0.5*(1.0 + math.erf(x / _SQRT2))

def bs_price(S, K, T_days, s_daily, is_call=True):
    """Calculates the theoretical price of an option (r=0, q=0)."""
    if S <= 0 or K <= 0: return 0.0
    
    # Intrinsic value if expired or zero volatility
    if T_days < 1e-9 or s_daily < 1e-9:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
        
    sq = s_daily * math.sqrt(T_days)
    d1 = (math.log(S/K) + 0.5 * s_daily * s_daily * T_days) / sq
    d2 = d1 - sq
    
    if is_call:
        return S * _ncdf(d1) - K * _ncdf(d2)
    else:
        return K * _ncdf(-d2) - S * _ncdf(-d1)

def bs_delta(S, K, T_days, s_daily, is_call=True):
    """Calculates the directional risk (Delta)."""
    if S <= 0 or K <= 0: return 0.0
    
    if T_days < 1e-9 or s_daily < 1e-9:
        if is_call:
            return 1.0 if S >= K else 0.0
        else:
            return -1.0 if S <= K else 0.0
            
    d1 = (math.log(S/K) + 0.5 * s_daily * s_daily * T_days) / (s_daily * math.sqrt(T_days))
    return _ncdf(d1) if is_call else _ncdf(d1) - 1.0

def bs_vega(S, K, T_days, s_daily):
    """Calculates volatility sensitivity (Vega). Identical for Calls and Puts."""
    if S <= 0 or K <= 0: return 0.0
    if T_days < 1e-9 or s_daily < 1e-9: return 0.0
    
    sq = s_daily * math.sqrt(T_days)
    d1 = (math.log(S/K) + 0.5 * s_daily * s_daily * T_days) / sq
    return S * _npdf(d1) * math.sqrt(T_days)

def bs_vanna(S, K, T_days, s_daily):
    """Calculates Vega sensitivity to spot (Vanna). Highly optimized."""
    if S <= 0 or K <= 0: return 0.0
    if T_days < 1e-9 or s_daily < 1e-9: return 0.0
    
    sq = s_daily * math.sqrt(T_days)
    d1 = (math.log(S/K) + 0.5 * s_daily * s_daily * T_days) / sq
    
    # Reuses Vega calculation logic mathematically
    return bs_vega(S, K, T_days, s_daily) * (1.0 - d1/sq) / S

def implied_vol(Price, S, K, T_days, is_call=True, s0=0.015):
    """
    Solves for Implied Volatility using Newton-Raphson with a Bisection fallback.
    s0 is seeded with 0.015 (approx 24% annual vol scaled to daily).
    """
    if S <= 0 or K <= 0: return None
    
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if Price <= intrinsic + 1e-5:
        return None  # Option is trading below intrinsic value
        
    s = s0
    for _ in range(30):
        p = bs_price(S, K, T_days, s, is_call)
        v = bs_vega(S, K, T_days, s)
        e = p - Price
        
        if abs(e) < 1e-4: return s
        if v < 1e-10: break
        
        # Bounded Newton-Raphson (Daily Volatility bounds: 0.05% to 30% per day)
        s = max(0.0005, min(0.3, s - e/v)) 
        
    # Bisection Fallback if Newton fails to converge
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
# ─────────────────────────────────────────────────────────────────────────────
# Order-book helpers
# ─────────────────────────────────────────────────────────────────────────────
def _best_bid(od):
    if not od.buy_orders:  return None, 0
    p = max(od.buy_orders); return p, od.buy_orders[p]

def _best_ask(od):
    if not od.sell_orders: return None, 0
    p = min(od.sell_orders); return p, od.sell_orders[p]


# ─────────────────────────────────────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────────────────────────────────────
class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {sym: [] for sym in SYMS.values()}
        result[UNDERLYING] = []
        conversions = 0

        # ── Restore state ─────────────────────────────────────────────
        try:    ps = json.loads(state.traderData) if state.traderData else {}
        except: ps = {}
        iv_seeds = ps.get("ivs", {str(K): 0.24 for K in ALL_STRIKES})
        t_decay  = float(ps.get("t", 1.0))    # signal strength
        prev_residuals = ps.get("pr", {})     # residuals from prior tick

        # ── Time, underlying ─────────────────────────────────────────
        tick   = int(state.timestamp)
        T_days = max((TTE_START_DAYS - tick / TICKS_PER_DAY), 1e-9)
        sqrt_T = math.sqrt(T_days)

        S_obs  = None
        vfe_bp: Optional[int] = None
        vfe_ap: Optional[int] = None
        if UNDERLYING in state.order_depths:
            od_v     = state.order_depths[UNDERLYING]
            bp_v, _  = _best_bid(od_v); ap_v, _ = _best_ask(od_v)
            if bp_v is not None and ap_v is not None:
                S_obs  = 0.5 * (bp_v + ap_v)
                vfe_bp = bp_v
                vfe_ap = ap_v
        S_eff = S_obs if S_obs is not None else 5250.0

        # ── Read book state for all strikes ──────────────────────────
        live_bids: Dict[int, Tuple[int, int]] = {}
        live_asks: Dict[int, Tuple[int, int]] = {}
        live_mids: Dict[int, float] = {}
        live_iv_obs:  Dict[int, float] = {}     # raw IV from market
        live_iv_adj:  Dict[int, float] = {}     # Lo-Wang adjusted

        for K in ALL_STRIKES:
            sym = SYMS[K]
            if sym not in state.order_depths: continue
            od = state.order_depths[sym]
            bp, bq = _best_bid(od)
            ap, aq = _best_ask(od)
            if bp is None or ap is None: continue
            live_bids[K] = (bp, bq)
            live_asks[K] = (ap, aq)
            mp = 0.5*(bp + ap)
            live_mids[K] = mp

            seed = float(iv_seeds.get(str(K), 0.24))
            iv = implied_vol(mp, S_eff, K, T_days, is_call=True, s0=seed)
            if iv is not None:
                iv_seeds[str(K)] = iv
                live_iv_obs[K] = iv
                live_iv_adj[K] = iv  #* LO_WANG_FACTOR

        # ── Regime filter: tight surface? ────────────────────────────
        sp_5200 = (live_asks[5200][0] - live_bids[5200][0]) if 5200 in live_bids and 5200 in live_asks else 999
        sp_5300 = (live_asks[5300][0] - live_bids[5300][0]) if 5300 in live_bids and 5300 in live_asks else 999
        tight_surface = (sp_5200 <= TIGHT_SPREAD_MAX) and (sp_5300 <= TIGHT_SPREAD_MAX)

        # Update signal strength
        if tight_surface:
            t_decay = 1.0
        else:
            t_decay = t_decay * DECAY_FACTOR
            if t_decay < 0.05:
                t_decay = 0.05            # floor (don't go fully off)

        regime_mult = TIGHT_MULT if tight_surface else LOOSE_MULT

        # ── Compute per-strike fitted IV, residual, Greeks ──────────
        residuals: Dict[int, float] = {}
        vegas:     Dict[int, float] = {}
        deltas:    Dict[int, float] = {}
        vannas:    Dict[int, float] = {}
        thresh:    Dict[int, float] = {}
        sides:     Dict[int, int]   = {}     # +1 BUY, -1 SELL, 0 NONE

        # Find ATM-most vega for size scaling
        max_vega = 1.0
        for K in SIGNAL_STRIKES:
            if K not in live_iv_adj: continue
            v = bs_vega(S_eff, K, T_days, live_iv_adj[K])
            if v > max_vega: max_vega = v

        for K in ALL_STRIKES:
            if K not in live_iv_adj: continue
            iv_a = live_iv_adj[K]
            iv_o = live_iv_obs[K]

            # Hard-coded smile fitted IV
            m       = math.log(K / S_eff) / sqrt_T
            iv_fit  = SMILE_A*m*m + SMILE_B*m + SMILE_C
            r       = iv_a - iv_fit          # residual on adjusted IV
            residuals[K] = r

            # Greeks computed at observed IV (for live sensitivity)
            vegas[K]  = bs_vega(S_eff, K, T_days, iv_o)
            deltas[K] = bs_delta(S_eff, K, T_days, iv_o, is_call=True)
            vannas[K] = bs_vanna(S_eff, K, T_days, iv_o)

            # Per-strike threshold
            half_sp = 0.0
            if K in live_bids and K in live_asks:
                half_sp = (live_asks[K][0] - live_bids[K][0]) / 2.0
            cost_thresh = half_sp / vegas[K] if vegas[K] > 1e-6 else 1e6
            thresh[K]   = max(cost_thresh, regime_mult * IV_NOISE_FLOOR)

            # Side: trade only signal strikes
            if K in SIGNAL_STRIKES and abs(r) > thresh[K]:
                # r > 0  : adjusted IV above smile  → market RICH  → SELL
                # r < 0  : adjusted IV below smile  → market CHEAP → BUY
                sides[K] = -1 if r > 0 else +1
            else:
                sides[K] = 0

        # ── Build target portfolio ───────────────────────────────────
        targets: Dict[int, int] = {K: 0 for K in ALL_STRIKES}

        # Provisional sizes weighted by ATM-ness and signal strength t
        for K in SIGNAL_STRIKES:
            if sides.get(K, 0) == 0: continue
            v = vegas.get(K, 0.0)
            atm_weight = v / max_vega if max_vega > 0 else 0.0
            size = int(round(SCALP_BASE * t_decay * atm_weight))
            size = max(0, min(VOUCHER_LIMIT, size))
            targets[K] = sides[K] * size

        # ── Vanna trim: bring net_vanna inside tolerance ─────────────
        net_vanna = sum(targets[K] * vannas.get(K, 0.0) for K in ALL_STRIKES)

        if abs(net_vanna) > VANNA_TOL:
            # Find the active position whose vanna contribution OPPOSES net_vanna
            # the most strongly per unit (so trimming it reduces |net_vanna| fastest)
            # but we trim the position that costs us the LEAST edge per unit.
            # Heuristic: trim the position with smallest |residual * vega|.
            best_K, best_cost = None, float('inf')
            for K in SIGNAL_STRIKES:
                if targets[K] == 0: continue
                # Does shrinking |targets[K]| by 1 reduce |net_vanna|?
                contrib = targets[K] * vannas.get(K, 0.0)
                # If contrib has same sign as net_vanna, reducing |targets[K]| reduces |net_vanna|.
                if (contrib > 0 and net_vanna > 0) or (contrib < 0 and net_vanna < 0):
                    cost_per_unit = abs(residuals.get(K, 0.0)) * vegas.get(K, 0.0)
                    if cost_per_unit < best_cost:
                        best_cost = cost_per_unit
                        best_K    = K
            if best_K is not None:
                vanna_K = vannas.get(best_K, 0.0)
                if abs(vanna_K) > 1e-9:
                    # Reduce |target| by amount needed to push net_vanna back to 0
                    qty_remove = net_vanna / vanna_K     # signed
                    new_t = targets[best_K] - int(round(qty_remove))
                    # Don't flip sign
                    if (targets[best_K] > 0 and new_t < 0) or \
                       (targets[best_K] < 0 and new_t > 0):
                        new_t = 0
                    new_t = max(-VOUCHER_LIMIT, min(VOUCHER_LIMIT, new_t))
                    targets[best_K] = new_t

        # ── Vanna hedge sleeve: neutralize via 5000 / 5100 / 5500 ────
        # Vannas (typical):  5000:-0.62, 5100:-0.98, 5500:+0.72
        # If net_vanna > 0  → need NEGATIVE vanna contribution → BUY 5100/5000
        #                     (long * negative vanna = negative)
        #                  → OR SHORT 5500 (short * positive vanna = negative)
        # If net_vanna < 0  → opposite
        positions = state.position
        vfe_pos   = positions.get(UNDERLYING, 0)   # kept for logging only

        # Recompute net_vanna after the trim above (signal-strike provisional)
        # plus EXISTING positions on hedge strikes.
        proj_net_vanna = sum(targets[K] * vannas.get(K, 0.0) for K in SIGNAL_STRIKES) \
                       + sum(positions.get(SYMS[K], 0) * vannas.get(K, 0.0)
                             for K in VANNA_HEDGE_STRIKES)

        # Spike detection on 5200/5300 — forces aggressive hedge crossing
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
            # Choose hedge strike based on what flattens vanna with smallest slippage
            # Sort hedge strikes by |vanna| descending (most efficient first).
            ranked_hedge = sorted(
                VANNA_HEDGE_STRIKES,
                key=lambda k: -abs(vannas.get(k, 0.0))
            )

            for K in ranked_hedge:
                if abs(proj_net_vanna) <= VANNA_TOL: break
                if K not in live_bids or K not in live_asks: continue
                vanna_K = vannas.get(K, 0.0)
                if abs(vanna_K) < 1e-6: continue

                pos_K = positions.get(SYMS[K], 0)
                bp, bq = live_bids[K]
                ap, aq = live_asks[K]
                if ap - bp < 2: continue

                # Direction needed: change in hedge_pos that REDUCES |net_vanna|
                # delta_pos * vanna_K should be opposite-signed to net_vanna.
                if proj_net_vanna > 0:
                    # need negative contribution → delta_pos has opposite sign to vanna_K
                    direction = -1 if vanna_K > 0 else +1
                else:
                    direction = +1 if vanna_K > 0 else -1

                # Quantity to fully flatten via this strike
                qty_full = int(round(abs(proj_net_vanna / vanna_K)))

                if direction > 0:
                    room = VOUCHER_LIMIT - pos_K
                else:
                    room = VOUCHER_LIMIT + pos_K
                if room <= 0: continue

                if spike_detected:
                    qty = min(qty_full, room, VANNA_HEDGE_AGGRESSIVE,
                              (-aq if direction > 0 and aq < 0 else
                                bq if direction < 0 and bq > 0 else
                               VANNA_HEDGE_AGGRESSIVE))
                    if qty <= 0: continue
                    if direction > 0:
                        # Cross: BUY at ask
                        result[SYMS[K]].append(Order(SYMS[K], ap, qty))
                    else:
                        # Cross: SELL at bid
                        result[SYMS[K]].append(Order(SYMS[K], bp, -qty))
                    proj_net_vanna += direction * qty * vanna_K
                else:
                    qty = min(qty_full, room, VANNA_HEDGE_PASSIVE)
                    if qty <= 0: continue
                    if direction > 0:
                        # Passive BUY at bid+1
                        price = bp + EDGE_INSIDE
                        if price < ap:
                            result[SYMS[K]].append(Order(SYMS[K], price, qty))
                            proj_net_vanna += qty * vanna_K
                    else:
                        # Passive SELL at ask-1
                        price = ap - EDGE_INSIDE
                        if price > bp:
                            result[SYMS[K]].append(Order(SYMS[K], price, -qty))
                            proj_net_vanna -= qty * vanna_K

        # ── Issue orders to move toward targets (LADDERED) ──────────
        # SIGNAL_STRIKES: aggressive cross + ladder remainder across LADDER_LEVELS
        # VANNA_HEDGE_STRIKES: orders already issued in vanna sleeve above
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
                # Need to BUY
                buy_room = VOUCHER_LIMIT - pos - buy_qty[sym]
                if buy_room <= 0: continue
                remaining = min(gap, buy_room)

                # CROSS at ask if signal strong
                if r_K < -th and aq < 0:
                    take = min(remaining, -aq)
                    if take > 0:
                        result[sym].append(Order(sym, ap, take))
                        buy_qty[sym] += take
                        remaining -= take

                # LADDER passive across bid+1, bid+2, bid+3 (bounded by ap-1)
                if remaining > 0:
                    max_inside = ap - 1                     # don't cross
                    issued = 0
                    for level_idx in range(LADDER_LEVELS):
                        if remaining - issued <= 0: break
                        price = bp + EDGE_INSIDE + level_idx
                        if price > max_inside: break
                        frac = LADDER_FRACS[level_idx] if level_idx < len(LADDER_FRACS) else 0.0
                        level_qty = int(round(remaining * frac))
                        if level_idx == LADDER_LEVELS - 1:
                            level_qty = remaining - issued  # absorb rounding into last level
                        level_qty = min(level_qty, remaining - issued)
                        if level_qty > 0:
                            result[sym].append(Order(sym, price, level_qty))
                            buy_qty[sym] += level_qty
                            issued += level_qty

            else:  # gap < 0 → SELL
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

        # # ── Delta hedge on VELVETFRUIT_EXTRACT: target net_delta = 250 ──────
        # # net delta = sum(option_pos * option_delta) + underlying_pos
        # # Gap > 0 → we are short delta → passive BUY at best_bid + 1
        # # Gap < 0 → we are long delta  → passive SELL at best_ask - 1
        # net_delta_opts = sum(
        #     positions.get(SYMS[K], 0) * deltas.get(K, 0.0) for K in ALL_STRIKES
        # )
        # net_delta = net_delta_opts + vfe_pos
        # delta_gap = DELTA_TARGET - net_delta

        # if abs(delta_gap) > DELTA_TOL and vfe_bp is not None and vfe_ap is not None:
        #     qty_raw = int(round(abs(delta_gap)))
        #     if delta_gap > 0:
        #         # Need to BUY underlying: passive bid inside spread
        #         buy_room = VFE_LIMIT - vfe_pos
        #         qty = max(0, min(qty_raw, buy_room))
        #         if qty > 0:
        #             price = vfe_bp + EDGE_INSIDE
        #             if price < vfe_ap:          # don't cross
        #                 result[UNDERLYING].append(Order(UNDERLYING, price, qty))
        #     else:
        #         # Need to SELL underlying: passive ask inside spread
        #         sell_room = VFE_LIMIT + vfe_pos
        #         qty = max(0, min(qty_raw, sell_room))
        #         if qty > 0:
        #             price = vfe_ap - EDGE_INSIDE
        #             if price > vfe_bp:          # don't cross
        #                 result[UNDERLYING].append(Order(UNDERLYING, price, -qty))

        # ── Final position-limit safety ──────────────────────────────
        for K in ALL_STRIKES:
            sym = SYMS[K]
            pos = positions.get(sym, 0)
            tot_b = sum(o.quantity  for o in result[sym] if o.quantity > 0)
            tot_s = sum(-o.quantity for o in result[sym] if o.quantity < 0)
            if pos + tot_b > VOUCHER_LIMIT or pos - tot_s < -VOUCHER_LIMIT:
                result[sym] = []

        # VFE underlying limit
        u_b = sum(o.quantity  for o in result[UNDERLYING] if o.quantity > 0)
        u_s = sum(-o.quantity for o in result[UNDERLYING] if o.quantity < 0)
        if vfe_pos + u_b > VFE_LIMIT or vfe_pos - u_s < -VFE_LIMIT:
            result[UNDERLYING] = []

        # ── Persist state ────────────────────────────────────────────
        traderData = json.dumps({
            "ivs": iv_seeds,
            "t": t_decay,
            "pr": {str(K): residuals.get(K, 0.0) for K in [5200, 5300]},
        })

        #── Logging ──────────────────────────────────────────────────
        # net_d = net_delta
        # net_v = sum(positions.get(SYMS[K], 0) * vegas.get(K, 0.0)  for K in ALL_STRIKES)
        # net_va = sum(positions.get(SYMS[K], 0) * vannas.get(K, 0.0) for K in ALL_STRIKES)

        # regime = "TIGHT" if tight_surface else "loose"
        # pos_str = " ".join(
        #     f"{K}:{positions.get(SYMS[K],0):+d}/{targets[K]:+d}" for K in ALL_STRIKES
        # )
        # pos_str = pos_str + f" VFE:{vfe_pos:+d}"
        # res_str = " ".join(
        #     f"{K}:{residuals.get(K,0)*1000:+.1f}m" for K in SIGNAL_STRIKES
        # )
        # n_orders = sum(len(result[s]) for s in result)
        
        # # DIAGNOSTIC: per-strike vega and position contribution
        # vega_contrib_str = " ".join(
        #     f"{K}:{positions.get(SYMS[K],0)*vegas.get(K,0.0):+.0f}" for K in SIGNAL_STRIKES[:3]
        # )
        
        # if tick % 20000 == 0:
        #     print(
        #         f"t={tick} T_days={T_days:.4f} S={S_eff:.0f} reg={regime} t_dec={t_decay:.2f} "
        #         f"sp52={sp_5200} sp53={sp_5300} "
        #         f"netΔ={net_d:.0f} netV={net_v:.0f} netVa={net_va:.1f} ord={n_orders} "
        #         f"v_contrib_sample=[{vega_contrib_str}] "
        #         f"res=[{res_str}] pos/tgt=[{pos_str}]"
        #      )
        # ── VANNA HEDGE SLEEVE TELEMETRY ─────────────────────────────
        # hedge_orders_by_strike = {K: [] for K in VANNA_HEDGE_STRIKES}
        # for K in VANNA_HEDGE_STRIKES:
        #     hedge_orders_by_strike[K] = result[SYMS[K]]

        # for K in VANNA_HEDGE_STRIKES:
        #     orders = hedge_orders_by_strike[K]
        #     if not orders: continue
        #     if K not in live_bids or K not in live_asks: continue
        #     bp, _ = live_bids[K]
        #     ap, _ = live_asks[K]
        #     mid = (bp + ap) / 2.0
        #     total_qty = 0
        #     total_slip = 0.0
        #     for order in orders:
        #         slip = abs(order.price - mid)
        #         qty = abs(order.quantity)
        #         total_qty += qty
        #         total_slip += slip * qty
        #     avg_slip = total_slip / total_qty if total_qty > 0 else 0.0
        #     print(
        #             f"t={tick} VANNA_HEDGE K={K} | qty={total_qty} | "
        #             f"avg_slip={avg_slip:.3f} XIREC | total_cost={total_slip:.1f} XIREC"
        #         )

        return result, conversions, traderData