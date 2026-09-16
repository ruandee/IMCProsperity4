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
M55_EDGE              = 2.5   # ticks Mark 55 pays/gives vs mid on every trade
M55_SIZE              = 5    # max qty per side (Mark 55 trades 3-8 units)
# Only fade when swing is dormant AND |pos| is within this of zero.
# Keeps inventory manageable; at limit=200 and balanced M55 flow, peak ~±130.
M55_ACTIVE_POS_LIMIT  = 130


@dataclass
class VelvetState:
    fair: float | None = None
    n_obs: int = 0
    # Mark 55 tally (reset each day)
    m55_bought:    int        = 0
    m55_sold:      int        = 0
    m55_cost:      float      = 0.0
    m55_last_bid:  int | None = None   # prices posted last tick — used to match fills
    m55_last_ask:  int | None = None


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

        # # ── tally M55 fills from LAST tick via own_trades ─────────────────────
        # # own_trades reflects executions from the previous timestamp, so we
        # # match against the M55 prices we stored last tick.
        # for t in state.own_trades.get(self.VELVET, []):
        #     if ps.m55_last_bid is not None and t.price == ps.m55_last_bid and t.quantity > 0:
        #         ps.m55_bought += t.quantity
        #         ps.m55_cost   += t.price * t.quantity
        #     elif ps.m55_last_ask is not None and t.price == ps.m55_last_ask and t.quantity < 0:
        #         ps.m55_sold += abs(t.quantity)
        #         ps.m55_cost -= t.price * abs(t.quantity)

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
        else: swing_target = pos  # hold — neither entry nor exit threshold met

        swing_sent = False
        delta = swing_target - pos
        is_entry = abs(swing_target) == MAX_POS_SWING  # True = entering, False = exiting to 0

        if delta > 0:
            if is_entry:                               # aggressive buy — sweep the ask
                qty = min(delta, abs(depth.sell_orders[best_ask]))
                if qty > 0:
                    orders.append(Order(self.VELVET, best_ask, int(qty)))
                    pos += qty; swing_sent = True
            elif pos < 0:                              # passive exit — post bid, let fills come to us
                orders.append(Order(self.VELVET, best_bid, delta))
                pos += delta; swing_sent = True
        elif delta < 0:
            if is_entry:                               # aggressive sell — hit the bid
                qty = min(-delta, depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(self.VELVET, best_bid, int(-qty)))
                    pos -= qty; swing_sent = True
            elif pos > 0:                              # passive exit — post ask, let fills come to us
                orders.append(Order(self.VELVET, best_ask, delta))
                pos += delta; swing_sent = True

        # # ── LAYER 2: Mark 55 fade ─────────────────────────────────────────────
        # # Post passive limit orders at exactly the prices Mark 55 always crosses.
        # # Skip if swing just sent aggressive orders this tick, or position is
        # # too large (swing about to take over anyway).
        # #
        # # FIX: was `abs(swing_target) == 0` — that only allowed M55 inside the
        # # tight EXIT_THR band.  The correct check is that swing is NOT actively
        # # pushing to max position, i.e. swing_target is not ±MAX_POS_SWING.
        swing_dormant = abs(swing_target) != MAX_POS_SWING
        m55_active = (
            not swing_sent
            and swing_dormant
            and abs(pos) <= M55_ACTIVE_POS_LIMIT
        )

        if m55_active:
            bid_p = round(mid - M55_EDGE)
            ask_p = round(mid + M55_EDGE)
            buy_cap  = self.LIMIT - pos
            sell_cap = self.LIMIT + pos
            bid_sz = min(M55_SIZE, buy_cap)
            ask_sz = min(M55_SIZE, sell_cap)
            if bid_sz > 0:
                orders.append(Order(self.VELVET, bid_p,  int(bid_sz)))
            if ask_sz > 0:
                orders.append(Order(self.VELVET, ask_p, -int(ask_sz)))
            # Store prices so next tick can match fills in own_trades
            ps.m55_last_bid = bid_p if bid_sz > 0 else None
            ps.m55_last_ask = ask_p if ask_sz > 0 else None
        else:
            ps.m55_last_bid = None
            ps.m55_last_ask = None

        # # ── end-of-day M55 tally ──────────────────────────────────────────────
        # if state.timestamp == 999000:
        #     net_inv   = ps.m55_bought - ps.m55_sold
        #     mtm_value = net_inv * mid
        #     realised  = -ps.m55_cost
        #     total_pnl = realised + mtm_value
        #     print(
        #         f"\n{'='*60}\n"
        #         f"[M55 DAY SUMMARY]\n"
        #         f"  Bought : {ps.m55_bought} units\n"
        #         f"  Sold   : {ps.m55_sold} units\n"
        #         f"  Net inv: {net_inv:+d} units (MtM @ {mid:.1f} = {mtm_value:+.1f})\n"
        #         f"  Cash P&L (realised): {realised:+.1f}\n"
        #         f"  Total P&L (incl MtM): {total_pnl:+.1f}\n"
        #         f"{'='*60}\n"
        #     )

        # print(f"[VELVET ts={state.timestamp}] mid={mid:.1f} fair={ps.fair:.2f} "
        #       f"dev={dev:+.1f} pos={pos} swing_tgt={swing_target} "
        #       f"m55={'on' if m55_active else 'off'} "
        #       f"m55_b={ps.m55_bought} m55_s={ps.m55_sold}")

        return orders