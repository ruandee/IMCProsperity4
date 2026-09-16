"""
Live OU-process options pricing engine

    1. OU forward: F = mu + (S - mu) * exp(-theta * T)
    2. WLS smile fit (Level, Skew, Convexity) with weights 
        w_i = (vega_i / max(vega)) / max(spread_i, FLOOR) multiplied 
        by a freshness mask (zero if quote unchanged > 4 steps / 400 timestamps).
        Sufficient stats X'WX and X'Wy are EWMA'd across ticks; coefficients
        are extracted via closed-form 3x3 inverse
    3. Kalman update against a random-walk prior. Measurement noise R
       inherits the EWMA WLS covariance, so the gain self-attenuates when
       live spreads widen (adaptive Kalman gain on real-time uncertainty).
    4. Wing clamp at +/- 1.5 * M_STD on log-moneyness. Boundary IV uses the
       full smile (skew-aware via the c1 term); beyond the boundary, IV is
       extrapolated linearly with the boundary slope (convexity frozen).
    5. Trade signal-strikes whose residual r = iv_obs - iv_fit exceeds the
       half-spread vega cost: r < 0 → BUY; r > 0 → SELL.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import math
import json
import numpy as np


# Seed filter
#   X0_AVG = [+1.175461e-02, +3.435218e-02, -6.033931e-01]
X0    = np.array([+1.0772156350e-02, +1.0911297429e-01, -1.9502481895e+00])  # Level, Skew, Convexity @ TTE=4.0
P0    = np.array([
    [+7.026582e-06, -1.481289e-04, +2.392721e-03],
    [-1.481289e-04, +3.414312e-03, -5.699942e-02],
    [+2.392721e-03, -5.699942e-02, +9.690831e-01],
])
M_STD          = 0.032102
OU_MU          = 5246.8661
OU_THETA       = 21.79498  # per day
TTE_START_DAYS = 4.0

# Engine config 
UNDERLYING     = "VELVETFRUIT_EXTRACT"
STRIKES        = [5000, 5100, 5200, 5300, 5400, 5500]
SYMS           = {K: f"VEV_{K}" for K in STRIKES}
SIGNAL_STRIKES = [5100, 5200, 5300, 5400]
TS_PER_DAY     = 999000
TTE_START_DAYS = 5.0
VOUCHER_LIMIT  = 300

STALE_TS       = 400
SPREAD_FLOOR   = 0.5
VEGA_FLOOR     = 1e-8
WING_SIGMA     = 1.5
EWMA_ALPHA     = 0.05         # EWMA on sufficient stats; half-life approx. 14 steps
WARMUP_NEFF    = 0.30         # need approx. 7 effective steps before trusting fit
KF_Q_DIAG      = np.array([1e-7, 1e-7, 1e-8])
KF_R_FLOOR     = np.array([1e-6, 1e-6, 1e-7])

CLAMP_LO       = -WING_SIGMA * M_STD
CLAMP_HI       =  WING_SIGMA * M_STD

SCALP_BASE     = 60
COST_DISCOUNT  = 0.4
NOISE_FLOOR_IV = 0.002

SQRT_2PI = math.sqrt(2.0 * math.pi)
_I3      = np.eye(3)


## BS scalar
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


## Closed-form 3x3 inverse
def inv3(M):
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


def best_bid(od):
    if not od.buy_orders:  return None, 0
    p = max(od.buy_orders); return p, od.buy_orders[p]


def best_ask(od):
    if not od.sell_orders: return None, 0
    p = min(od.sell_orders); return p, od.sell_orders[p]


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {sym: [] for sym in SYMS.values()}
        conversions = 0

        ## Restore persisted state 
        try:    ps = json.loads(state.traderData) if state.traderData else {}
        except: ps = {}

        x      = np.array(ps.get("x",  X0.tolist()),                dtype=float)
        P      = np.array(ps.get("P",  P0.flatten().tolist()),      dtype=float).reshape(3, 3)
        S_AA   = np.array(ps.get("AA", [0.0] * 9),                  dtype=float).reshape(3, 3)
        S_Ay   = np.array(ps.get("Ay", [0.0] * 3),                  dtype=float)
        n_eff  = float(ps.get("ne", 0.0))
        ivs    = ps.get("ivs", {str(K): 0.015 for K in STRIKES})
        last_q = ps.get("q",   {})
        last_t = ps.get("lts", {})

        ## Time spot OU fwd
        tick   = int(state.timestamp)
        T_days = max(TTE_START_DAYS - tick / TS_PER_DAY, 1e-9)

        S_obs = None
        if UNDERLYING in state.order_depths:
            od = state.order_depths[UNDERLYING]
            bp, _ = best_bid(od); ap, _ = best_ask(od)
            if bp is not None and ap is not None:
                S_obs = 0.5 * (bp + ap)
        if S_obs is None:
            return result, conversions, state.traderData or ""

        F = OU_MU + (S_obs - OU_MU) * math.exp(-OU_THETA * T_days)

        ## IV vega spread freshness
        K_a, iv_a, v_a, sp_a, fr_a = [], [], [], [], []
        live_bids: Dict[int, tuple] = {}
        live_asks: Dict[int, tuple] = {}

        for K in STRIKES:
            sym = SYMS[K]
            if sym not in state.order_depths: continue
            od = state.order_depths[sym]
            bp, bq = best_bid(od); ap, aq = best_ask(od)
            if bp is None or ap is None: continue
            mid    = 0.5 * (bp + ap)
            spread = max(ap - bp, SPREAD_FLOOR) #hard denom floor

            seed = float(ivs.get(str(K), 0.015))
            iv   = implied_vol(mid, F, K, T_days, seed)
            if iv is None: continue
            ivs[str(K)] = iv

            v = max(bs_vega(F, K, T_days, iv), VEGA_FLOOR)

            ## Freshness: track when current (bp,ap) first appeared
            kk  = str(K)
            cur = [bp, ap]
            if last_q.get(kk) != cur:
                last_q[kk] = cur
                last_t[kk] = tick
            age   = tick - int(last_t.get(kk, tick))
            fresh = 1.0 if age <= STALE_TS else 0.0

            K_a.append(K); iv_a.append(iv); v_a.append(v)
            sp_a.append(spread); fr_a.append(fresh)
            live_bids[K] = (bp, bq); live_asks[K] = (ap, aq)

        ## Build current-tick X, w; EWMA-update sufficient stats
        if len(K_a) >= 3:
            K_np  = np.asarray(K_a,  dtype=float)
            iv_np = np.asarray(iv_a, dtype=float)
            v_np  = np.asarray(v_a,  dtype=float)
            sp_np = np.asarray(sp_a, dtype=float)
            fr_np = np.asarray(fr_a, dtype=float)

            m_np  = np.log(K_np / F)
            w_np  = (v_np / v_np.max()) / sp_np * fr_np   # zero-out stale

            X     = np.column_stack([np.ones_like(m_np), m_np, m_np * m_np])
            Xw    = X * w_np[:, None]
            AA    = Xw.T @ X
            Ay    = Xw.T @ iv_np

            a_ew  = EWMA_ALPHA
            S_AA  = (1.0 - a_ew) * S_AA + a_ew * AA
            S_Ay  = (1.0 - a_ew) * S_Ay + a_ew * Ay
            n_eff = (1.0 - a_ew) * n_eff + a_ew * (1.0 if w_np.any() else 0.0)
        else:
            m_np  = None  # nothing to fit this tick

        ## WLS solve + adaptive Kalman update (only after warm-up)
        residuals: Dict[int, float] = {}
        if n_eff > WARMUP_NEFF and m_np is not None:
            cov_raw  = inv3(S_AA) # (X'WX)^{-1}
            beta_raw = cov_raw @ S_Ay # raw smile coeffs

            # Measurement noise tracks live spread uncertainty
            R_diag = np.maximum(np.diag(cov_raw), KF_R_FLOOR)
            R      = np.diag(R_diag)

            P_pred = P + np.diag(KF_Q_DIAG)
            S_inn  = P_pred + R
            K_g    = P_pred @ inv3(S_inn) # Kalman gain
            x      = x + K_g @ (beta_raw - x)

            I_KH   = _I3 - K_g #Joseph form?
            P      = I_KH @ P_pred @ I_KH.T + K_g @ R @ K_g.T
        else:
            # Predict-only: covariance grows by Q
            P = P + np.diag(KF_Q_DIAG)

        ## Wing-clamped fitted IV
        c0, c1, c2 = float(x[0]), float(x[1]), float(x[2])
        sig_lo   = c0 + c1 * CLAMP_LO + c2 * CLAMP_LO * CLAMP_LO
        sig_hi   = c0 + c1 * CLAMP_HI + c2 * CLAMP_HI * CLAMP_HI
        slope_lo = c1 + 2.0 * c2 * CLAMP_LO
        slope_hi = c1 + 2.0 * c2 * CLAMP_HI

        def fit_iv(m):
            if m < CLAMP_LO: return sig_lo + slope_lo * (m - CLAMP_LO)
            if m > CLAMP_HI: return sig_hi + slope_hi * (m - CLAMP_HI)
            return c0 + c1 * m + c2 * m * m

        ## Residual edge logic
        if m_np is not None and n_eff > WARMUP_NEFF:
            v_max = float(np.asarray(v_a).max())
            for K, iv, m, v, spread in zip(K_a, iv_a, m_np, v_a, sp_a):
                iv_f = fit_iv(float(m))
                r    = iv - iv_f
                residuals[K] = r
                if K not in SIGNAL_STRIKES: continue

                cost_iv = (spread * 0.5) / v
                thresh  = max(cost_iv * COST_DISCOUNT, NOISE_FLOOR_IV)
                if abs(r) < thresh: continue

                sym = SYMS[K]
                pos = state.position.get(sym, 0)
                bp, bq = live_bids[K]; ap, aq = live_asks[K]

                size = int(round(SCALP_BASE * (v / v_max)))
                if size <= 0: continue

                if r < 0:                                  # CHEAP → BUY at ask
                    room = VOUCHER_LIMIT - pos
                    qty  = min(size, room, max(-aq, 0))
                    if qty > 0:
                        result[sym].append(Order(sym, ap, qty))
                else:                                      # RICH → SELL at bid
                    room = VOUCHER_LIMIT + pos
                    qty  = min(size, room, max(bq, 0))
                    if qty > 0:
                        result[sym].append(Order(sym, bp, -qty))

        ## Position safety check
        for K in STRIKES:
            sym = SYMS[K]
            pos = state.position.get(sym, 0)
            tb  = sum(o.quantity  for o in result[sym] if o.quantity > 0)
            ts  = sum(-o.quantity for o in result[sym] if o.quantity < 0)
            if pos + tb > VOUCHER_LIMIT or pos - ts < -VOUCHER_LIMIT:
                result[sym] = []

        ##Persist 
        traderData = json.dumps({
            "x":   x.tolist(),
            "P":   P.flatten().tolist(),
            "AA":  S_AA.flatten().tolist(),
            "Ay":  S_Ay.tolist(),
            "ne":  n_eff,
            "ivs": ivs,
            "q":   last_q,
            "lts": last_t,
        })

        return result, conversions, traderData