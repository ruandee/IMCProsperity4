from datamodel import OrderDepth, TradingState, Order
from dataclasses import dataclass, asdict, field
from typing import List
import json, math

# ── swing params (unchanged) ──────────────────────────────────────────────────
FAIR_ANCHOR    = 5247.65
FAIR_DRIFT_CAP = 200.0
FAIR_EMA_ALPHA = 2e-6
ENTRY_THR      = 12.0
EXIT_THR       = 3.0
MAX_POS_SWING  = 200
MIN_HISTORY    = 50

# ── Mark 55 fade params ───────────────────────────────────────────────────────
M55_EDGE             = 2.5   # ticks Mark 55 pays/gives vs mid on every trade
M55_SIZE             = 6     # max qty per side once M55 is confirmed (trades 3-8 units)
M55_ACTIVE_POS_LIMIT = 130   # suppress fade when |pos| exceeds this

# ── Mark 55 detection params ──────────────────────────────────────────────────
# Behavioral fingerprint: M55 always crosses passive orders at exactly
# mid ± M55_EDGE with a fill qty in [M55_MIN_QTY, M55_MAX_QTY].
# We keep a confidence score that rises on confirmed fills and decays
# every tick that passes without one.  Only at/above threshold do we
# scale from probe size up to full fade size.
M55_MIN_QTY            = 3    # M55 never trades fewer than this
M55_MAX_QTY            = 8    # M55 never trades more than this
M55_PROBE_SIZE         = 1    # small qty posted while unconfirmed (generates signal)
M55_CONFIDENCE_PER_FILL = 2   # score added per confirmed fill
M55_CONFIDENCE_DECAY   = 1    # score lost per tick with no fill
M55_CONFIDENCE_MAX     = 10   # hard ceiling
M55_CONFIDENCE_THRESH  = 4    # score needed to go full size (≈ 2 back-to-back fills)


@dataclass
class VelvetState:
    fair: float | None = None
    n_obs: int = 0
    # ── Mark 55 tally (reset each day) ────────────────────────────────────────
    m55_bought:     int        = 0
    m55_sold:       int        = 0
    m55_cost:       float      = 0.0
    m55_last_bid:   int | None = None   # price posted last tick (bid side)
    m55_last_ask:   int | None = None   # price posted last tick (ask side)
    # ── Mark 55 detection state ───────────────────────────────────────────────
    # confidence rises when own_trades confirms the M55 fill signature and
    # decays passively so detection lapses if M55 goes quiet.
    m55_confidence: int        = 0
    m55_detected:   bool       = False


@dataclass
class SavedState:
    velvet: VelvetState = field(default_factory=lambda: VelvetState())

    @staticmethod
    def load(s):
        if not s:
            return SavedState()
        try:
            d = json.loads(s)
            return SavedState(velvet=VelvetState(**d.get("velvet", {})))
        except Exception:
            return SavedState()

    def dump(self):
        return json.dumps(asdict(self))


class Trader:
    VELVET = "VELVETFRUIT_EXTRACT"
    LIMIT  = 200

    def run(self, state: TradingState):
        saved = SavedState.load(state.traderData)
        orders = {self.VELVET: self._trade(state, saved)}
        return orders, 0, saved.dump()

    def _trade(self, state: TradingState, saved: SavedState) -> List[Order]:
        ps     = saved.velvet
        orders = []

        # ── STEP 1: tally M55 fills from LAST tick via own_trades ─────────────
        # own_trades reflects executions from the *previous* timestamp, so we
        # match against the M55 prices we stored last tick.
        #
        # Fingerprint check: price must match exactly AND fill qty must be in
        # the range M55 actually trades.  This filters out coincidental fills
        # from other participants crossing at the same level.
        got_m55_fill = False
        for t in state.own_trades.get(self.VELVET, []):
            qty = t.quantity  # positive = we bought, negative = we sold
            # Buy-side fill: we posted a bid at m55_last_bid and got hit
            if (
                ps.m55_last_bid is not None
                and t.price == ps.m55_last_bid
                and qty > 0
                and M55_MIN_QTY <= qty <= M55_MAX_QTY
            ):
                ps.m55_bought += qty
                ps.m55_cost   += t.price * qty
                got_m55_fill   = True
            # Sell-side fill: we posted an ask at m55_last_ask and got lifted
            elif (
                ps.m55_last_ask is not None
                and t.price == ps.m55_last_ask
                and qty < 0
                and M55_MIN_QTY <= abs(qty) <= M55_MAX_QTY
            ):
                ps.m55_sold += abs(qty)
                ps.m55_cost -= t.price * abs(qty)
                got_m55_fill  = True

        # ── STEP 2: update confidence score ───────────────────────────────────
        # Rise on confirmed fill, decay passively so presence lapses when quiet.
        if got_m55_fill:
            ps.m55_confidence = min(
                M55_CONFIDENCE_MAX,
                ps.m55_confidence + M55_CONFIDENCE_PER_FILL,
            )
        else:
            ps.m55_confidence = max(0, ps.m55_confidence - M55_CONFIDENCE_DECAY)

        ps.m55_detected = ps.m55_confidence >= M55_CONFIDENCE_THRESH

        # ── market data ───────────────────────────────────────────────────────
        depth = state.order_depths.get(self.VELVET)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return orders

        pos      = state.position.get(self.VELVET, 0)
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        if best_bid >= best_ask:
            return orders
        mid = (best_bid + best_ask) / 2.0

        # ── fair value EMA ────────────────────────────────────────────────────
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

        # ── LAYER 1: swing ────────────────────────────────────────────────────
        if   dev >  ENTRY_THR: swing_target = -MAX_POS_SWING
        elif dev < -ENTRY_THR: swing_target = +MAX_POS_SWING
        elif abs(dev) < EXIT_THR: swing_target = 0
        else: swing_target = 0   # neutral band — begin passive exit immediately

        swing_sent = False
        delta = swing_target - pos
        is_entry = abs(swing_target) == MAX_POS_SWING

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

        # ── LAYER 2: Mark 55 fade (detection-gated) ───────────────────────────
        # Gate logic:
        #   swing_dormant — swing is not actively pushing to max position
        #   m55_layer_ok  — position headroom is available
        #
        # Size logic (two modes):
        #   PROBE  (m55_detected=False) — post M55_PROBE_SIZE (1 unit) to
        #          generate own_trades signal at the exact M55 price levels.
        #          Low risk, purely for fingerprinting.
        #   FULL   (m55_detected=True)  — post M55_SIZE (6 units) to capture
        #          the full ~2.5-tick edge on every M55 crossing.
        #
        # We never suppress posting entirely (except when swing is aggressive),
        # because the probe is what gives us detection data in the first place.
        swing_dormant = abs(swing_target) != MAX_POS_SWING
        m55_layer_ok  = (
            not swing_sent
            and swing_dormant
            and abs(pos) <= M55_ACTIVE_POS_LIMIT
        )

        if m55_layer_ok:
            bid_p    = round(mid - M55_EDGE)
            ask_p    = round(mid + M55_EDGE)
            buy_cap  = self.LIMIT - pos
            sell_cap = self.LIMIT + pos

            # Choose size based on detected state
            target_size = M55_SIZE if ps.m55_detected else M55_PROBE_SIZE

            bid_sz = min(target_size, buy_cap)
            ask_sz = min(target_size, sell_cap)

            if bid_sz > 0:
                orders.append(Order(self.VELVET, bid_p,  int(bid_sz)))
            if ask_sz > 0:
                orders.append(Order(self.VELVET, ask_p, -int(ask_sz)))

            # Store posted prices so next tick can attribute own_trades fills
            ps.m55_last_bid = bid_p if bid_sz > 0 else None
            ps.m55_last_ask = ask_p if ask_sz > 0 else None
        else:
            # Not posting — clear stored prices so stale fills aren't credited
            ps.m55_last_bid = None
            ps.m55_last_ask = None

        # ── end-of-day M55 tally ──────────────────────────────────────────────
        if state.timestamp == 999000:
            net_inv   = ps.m55_bought - ps.m55_sold
            mtm_value = net_inv * mid
            realised  = -ps.m55_cost
           # total_pnl = realised + mtm_value
            

        # print(f"[VELVET ts={state.timestamp}] mid={mid:.1f} fair={ps.fair:.2f} "
        #       f"dev={dev:+.1f} pos={pos} swing_tgt={swing_target} "
        #       f"m55_conf={ps.m55_confidence} m55_det={ps.m55_detected} "
        #       f"m55_b={ps.m55_bought} m55_s={ps.m55_sold}")

        return orders