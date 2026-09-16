"""
═══════════════════════════════════════════════════════════════════════════════
IMC Prosperity 4 – Round 3
Version G: Hard-Coded IV Smile + Regime-Conditional Vega/Vanna-Neutral Scalp
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Detect IV deviations against a FIXED, offline-fit parabola.  Trade only when
the microstructure regime supports profitable execution.  Stay roughly
vanna-neutral (small net Δvega per Δspot).  Net vega may run negative; that's
acceptable.  Hedge with deep-ITM low-vega vouchers (VEV_4000, VEV_4500), NOT
the underlying.

HARD-CODED SMILE  (offline-fit on 3M ticks)
───────────────────────────────────────────
        IV(m) = 0.185625·m² − 0.373365·m + 0.214202
where   m = ln(K/S) / √T

LO-WANG ADJUSTMENT  (multiplicative on σ, not σ²)
─────────────────────────────────────────────────
        σ_adj = σ_observed · 0.82612

The observed IV from market is multiplied by 0.82612 BEFORE comparing to the
fitted IV.  This makes the residuals match the smile in the offline fit.

REGIME FILTER  (microstructure gating)
──────────────────────────────────────
        tight_surface  =  (spread_5200 ≤ 2) AND (spread_5300 ≤ 2)

Time-decay signal strength  t ∈ [0, 1] :
        if tight_surface:   t = 1.0          (instant reset)
        else:               t *= 0.95        (per-tick decay)

In the TIGHT regime the bot we're racing is active → fills are cheap and
mean-reversion is fast → 2× edge per event.  In the LOOSE regime, the bot
backs off → wider spreads and stale signals → trade smaller / not at all.

ORDER SIZING
────────────
        size_k = SCALP_BASE · t · (vega_k / vega_max)
                                  └────── ATM-weighting ──────┘

The closer to ATM, the larger the size: ATM strikes have the highest vega
per contract so they offer the best edge per unit of inventory risk.

THRESHOLDS  (per-strike, regime-dependent)
──────────────────────────────────────────
        cost_thresh_k    =  half_spread_k / vega_k        (must cover spread)
        regime_mult      =  0.5  if tight_surface else 2.0
        thresh_k         =  max(cost_thresh_k, regime_mult · IV_NOISE_FLOOR)

Tight regime → trade smaller deviations (cleaner signal).
Loose regime → require larger deviations (noisier signal).

VANNA-NEUTRAL ALLOCATION
────────────────────────
For each signal-active strike compute provisional target = side_k · size_k.
Then trim/balance to keep:
        |net_vanna| < VANNA_TOL

Vanna for a call:   ν_K = vega_K · (1 − d1_K / (σ √T)) / S
                          (positive for OTM calls, negative for deep ITM)

Net delta is allowed to drift, then partially absorbed by VEV_4000/VEV_4500
(δ ≈ 1, vega ≈ 0).  These two vouchers are NOT scalped — they exist purely
to bleed off accumulated delta without spending vega budget.

POSITION LIMITS
   Each voucher  ±300.   VFE not traded.
═══════════════════════════════════════════════════════════════════════════════
"""

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

# Strikes used for IV-deviation scalping
SIGNAL_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]

# Deep-ITM vouchers (delta ≈ 1, vega ≈ 0) – pure delta sleeve
HEDGE_STRIKES  = [4000, 4500]

ALL_STRIKES    = HEDGE_STRIKES + SIGNAL_STRIKES
SYMS           = {K: f"VEV_{K}" for K in ALL_STRIKES}

VOUCHER_LIMIT  = 300

# ── Hard-coded IV smile (offline fit on 3M ticks) ────────────────────────
SMILE_A = 0.185625
SMILE_B = -0.373365
SMILE_C = 0.214202

# ── Lo-Wang multiplicative adjustment ────────────────────────────────────
LO_WANG_FACTOR = 0.82612          # σ_adj = σ_observed · 0.82612

# ── Regime filter ────────────────────────────────────────────────────────
TIGHT_SPREAD_MAX = 2              # spread ≤ 2 on both 5200 and 5300
DECAY_FACTOR     = 0.95           # per-tick decay of signal strength

# ── Sizing ───────────────────────────────────────────────────────────────
SCALP_BASE       = 175             # base contracts per scalp at full t and ATM
EDGE_INSIDE      = 1              # ticks inside best bid/ask for passive

# ── Thresholds ───────────────────────────────────────────────────────────
IV_NOISE_FLOOR   = 0.001          # 10 bps IV minimum residual to consider
TIGHT_MULT       = 0.5            # tight regime: lower threshold
LOOSE_MULT       = 2.0            # loose regime: higher threshold

# ── Vanna neutrality tolerance ──────────────────────────────────────────
VANNA_TOL        = 50.0

# ── Delta sleeve sizing (uses 4000/4500 to bleed delta) ─────────────────
DELTA_TOL        = 80.0
DELTA_SLEEVE_QTY = 30             # contracts per tick into 4000/4500 to flatten

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────
_SQRT2   = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

def _npdf(x): return math.exp(-0.5*x*x) / _SQRT2PI
def _ncdf(x): return 0.5*(1.0 + math.erf(x / _SQRT2))

def bs_price(S, K, T, s):
    if T < 1e-9 or s < 1e-9: return max(S - K, 0.0)
    sq = s*math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*s*s*T) / sq
    return S*_ncdf(d1) - K*_ncdf(d1 - sq)

def bs_delta(S, K, T, s):
    if T < 1e-9 or s < 1e-9: return 1.0 if S >= K else 0.0
    d1 = (math.log(S/K) + 0.5*s*s*T) / (s*math.sqrt(T))
    return _ncdf(d1)

def bs_vega(S, K, T, s):
    if T < 1e-9 or s < 1e-9: return 0.0
    sq = s*math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*s*s*T) / sq
    return S*_npdf(d1)*math.sqrt(T)

def bs_vanna(S, K, T, s):
    """∂²V/(∂σ ∂S) = vega · (1 − d1/(σ√T)) / S"""
    if T < 1e-9 or s < 1e-9: return 0.0
    sq = s*math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*s*s*T) / sq
    return bs_vega(S, K, T, s) * (1.0 - d1/sq) / S

def implied_vol(C, S, K, T, s0=0.30):
    intrinsic = max(S - K, 0.0)
    if C <= intrinsic + 1e-5:
        return None
    s = s0
    for _ in range(30):
        p = bs_price(S, K, T, s)
        v = bs_vega(S, K, T, s)
        e = p - C
        if abs(e) < 1e-4: return s
        if v < 1e-10: break
        s = max(0.01, min(5.0, s - e/v))
    lo, hi = 0.01, 5.0
    for _ in range(60):
        m = 0.5*(lo+hi)
        if bs_price(S, K, T, m) < C: lo = m
        else: hi = m
        if hi - lo < 1e-6: return m
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

        # ── Time, underlying ─────────────────────────────────────────
        tick    = int(state.timestamp)
        T_years = max((TTE_START_DAYS - tick / TICKS_PER_DAY) / 365.0, 1e-9)
        sqrt_T  = math.sqrt(T_years)

        S_obs = None
        if UNDERLYING in state.order_depths:
            od_v  = state.order_depths[UNDERLYING]
            bp, _ = _best_bid(od_v); ap, _ = _best_ask(od_v)
            if bp is not None and ap is not None:
                S_obs = 0.5 * (bp + ap)
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
            iv = implied_vol(mp, S_eff, K, T_years, s0=seed)
            if iv is not None:
                iv_seeds[str(K)] = iv
                live_iv_obs[K] = iv
                live_iv_adj[K] = iv * LO_WANG_FACTOR

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
            v = bs_vega(S_eff, K, T_years, live_iv_adj[K])
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
            vegas[K]  = bs_vega(S_eff, K, T_years, iv_o)
            deltas[K] = bs_delta(S_eff, K, T_years, iv_o)
            vannas[K] = bs_vanna(S_eff, K, T_years, iv_o)

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

        # ── Delta sleeve: trim |Δ| using 4000/4500 ──────────────────
        # Compute net delta INCLUDING current positions (not just targets,
        # because hedge sleeve is set via target position, then orders move
        # toward it).
        positions = state.position
        cur_net_delta = sum(positions.get(SYMS[K], 0) * deltas.get(K, 0.0)
                            for K in ALL_STRIKES)
        # If signals plus current positions imply big delta, set hedge target
        # to absorb it.
        proj_net_delta = sum(targets[K] * deltas.get(K, 0.0) for K in SIGNAL_STRIKES) \
                       + sum(positions.get(SYMS[K], 0) * deltas.get(K, 0.0)
                             for K in HEDGE_STRIKES)
        # Note: we use TARGET delta from signals + CURRENT delta from hedges,
        # because hedges are what we adjust.

        if abs(proj_net_delta) > DELTA_TOL:
            # Use the deeper-ITM (closer-to-1 delta) one first: 4000
            # Negative proj_net_delta → need MORE delta → BUY hedge
            # Positive proj_net_delta → need LESS delta → SELL hedge
            for K in [4000, 4500]:        # 4000 first (highest delta)
                if abs(proj_net_delta) <= DELTA_TOL: break
                d_h = deltas.get(K, 1.0)
                if d_h < 0.5: continue    # not effective if delta low
                pos_h = positions.get(SYMS[K], 0)
                if proj_net_delta > 0:
                    # short hedge to subtract delta
                    qty = min(DELTA_SLEEVE_QTY, VOUCHER_LIMIT + pos_h,
                              int(abs(proj_net_delta) / d_h))
                    if qty > 0:
                        targets[K] = pos_h - qty
                        proj_net_delta -= qty * d_h
                else:
                    qty = min(DELTA_SLEEVE_QTY, VOUCHER_LIMIT - pos_h,
                              int(abs(proj_net_delta) / d_h))
                    if qty > 0:
                        targets[K] = pos_h + qty
                        proj_net_delta += qty * d_h

        # ── Issue orders to move toward targets ──────────────────────
        buy_qty:  Dict[str, int] = {sym: 0 for sym in result}
        sell_qty: Dict[str, int] = {sym: 0 for sym in result}

        for K in ALL_STRIKES:
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
                qty = min(gap, buy_room)

                # CROSS aggressively if residual is strongly negative
                # (market significantly cheaper than smile)
                if r_K < -th and aq < 0:
                    take = min(qty, -aq)
                    if take > 0:
                        result[sym].append(Order(sym, ap, take))
                        buy_qty[sym] += take
                        qty -= take

                # Passive bid+1 for remainder
                if qty > 0:
                    price = bp + EDGE_INSIDE
                    if price < ap:
                        result[sym].append(Order(sym, price, qty))
                        buy_qty[sym] += qty

            else:  # gap < 0 → SELL
                sell_room = VOUCHER_LIMIT + pos - sell_qty[sym]
                if sell_room <= 0: continue
                qty = min(-gap, sell_room)

                if r_K > th and bq > 0:
                    take = min(qty, bq)
                    if take > 0:
                        result[sym].append(Order(sym, bp, -take))
                        sell_qty[sym] += take
                        qty -= take

                if qty > 0:
                    price = ap - EDGE_INSIDE
                    if price > bp:
                        result[sym].append(Order(sym, price, -qty))
                        sell_qty[sym] += qty

        # ── Final position-limit safety ──────────────────────────────
        for K in ALL_STRIKES:
            sym = SYMS[K]
            pos = positions.get(sym, 0)
            tot_b = sum(o.quantity  for o in result[sym] if o.quantity > 0)
            tot_s = sum(-o.quantity for o in result[sym] if o.quantity < 0)
            if pos + tot_b > VOUCHER_LIMIT or pos - tot_s < -VOUCHER_LIMIT:
                result[sym] = []

        # ── Persist state ────────────────────────────────────────────
        traderData = json.dumps({"ivs": iv_seeds, "t": t_decay})

        # ── Logging ──────────────────────────────────────────────────
        # net_d = sum(positions.get(SYMS[K], 0) * deltas.get(K, 0.0) for K in ALL_STRIKES)
        # net_v = sum(positions.get(SYMS[K], 0) * vegas.get(K, 0.0)  for K in ALL_STRIKES)
        # net_va = sum(positions.get(SYMS[K], 0) * vannas.get(K, 0.0) for K in ALL_STRIKES)

        # regime = "TIGHT" if tight_surface else "loose"
        # pos_str = " ".join(
        #     f"{K}:{positions.get(SYMS[K],0):+d}/{targets[K]:+d}" for K in ALL_STRIKES
        # )
        # res_str = " ".join(
        #     f"{K}:{residuals.get(K,0)*1000:+.1f}m" for K in SIGNAL_STRIKES
        # )
        # n_orders = sum(len(result[s]) for s in result)
        # print(
        #     f"t={tick} S={S_eff:.0f} reg={regime} t_dec={t_decay:.2f} "
        #     f"sp52={sp_5200} sp53={sp_5300} "
        #     f"netΔ={net_d:.0f} netV={net_v:.0f} netVa={net_va:.1f} ord={n_orders} "
        #     f"res=[{res_str}] pos/tgt=[{pos_str}]"
        # )

        return result, conversions, traderData