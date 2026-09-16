from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


PRODUCTS = [
    "PANEL_1X2",
    "PANEL_1X4",
    "PANEL_2X2",
    "PANEL_2X4",
    "PANEL_4X4",
]

# Kept deliberately conservative. If the true exchange limit is larger, this
# still avoids building positions that a simple market maker cannot work out of.
SOFT_POS_LIMIT = 8
HARD_POS_LIMIT = 10
BASE_ORDER_QTY = 2

EMA_ALPHA = 0.18
VOL_WINDOW = 40

MIN_QUOTE_EDGE = 2
TAKE_EDGE = 4
INV_SKEW_PER_UNIT = 0.45


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    var = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


class Trader:

    def bid(self):
        return 15

    @staticmethod
    def _book_snapshot(od: Optional[OrderDepth]) -> Optional[dict]:
        if od is None or not od.buy_orders or not od.sell_orders:
            return None

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        if best_bid >= best_ask:
            return None

        bid_vol = sum(v for v in od.buy_orders.values() if v > 0)
        ask_vol = sum(-v for v in od.sell_orders.values() if v < 0)
        if bid_vol <= 0 or ask_vol <= 0:
            return None

        bid_vwap = sum(px * v for px, v in od.buy_orders.items() if v > 0) / bid_vol
        ask_vwap = sum(px * -v for px, v in od.sell_orders.items() if v < 0) / ask_vol
        weighted_mid = (bid_vwap + ask_vwap) / 2.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": best_ask - best_bid,
            "weighted_mid": weighted_mid,
        }

    @staticmethod
    def _load_state(raw: str) -> dict:
        default = {
            "ema": {p: None for p in PRODUCTS},
            "mid_hist": {p: [] for p in PRODUCTS},
        }
        if not raw:
            return default

        try:
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                return default
        except Exception:
            return default

        loaded.setdefault("ema", {})
        loaded.setdefault("mid_hist", {})
        for p in PRODUCTS:
            loaded["ema"].setdefault(p, None)
            loaded["mid_hist"].setdefault(p, [])
        return loaded

    @staticmethod
    def _save_state(data: dict) -> str:
        for p in PRODUCTS:
            data["mid_hist"][p] = data["mid_hist"][p][-VOL_WINDOW:]
        return json.dumps(data)

    @staticmethod
    def _fair_value(product: str, book_mid: float, data: dict) -> float:
        prev_ema = data["ema"].get(product)
        if prev_ema is None:
            ema = book_mid
        else:
            ema = EMA_ALPHA * book_mid + (1.0 - EMA_ALPHA) * float(prev_ema)

        data["ema"][product] = ema
        data["mid_hist"][product].append(book_mid)
        data["mid_hist"][product] = data["mid_hist"][product][-VOL_WINDOW:]

        # Blend the live book with the rolling estimate. This is a small
        # stabilizer, not a predictive model.
        return 0.65 * book_mid + 0.35 * ema

    @staticmethod
    def _quote_edge(product: str, spread: float, data: dict) -> int:
        mids = data["mid_hist"].get(product, [])
        diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        tick_vol = _std(diffs)

        edge = max(MIN_QUOTE_EDGE, 0.35 * spread, 0.50 * tick_vol)
        return max(1, int(round(edge)))

    @staticmethod
    def _take_mispriced_book(
        product: str,
        fair: float,
        pos: int,
        od: OrderDepth,
        orders: List[Order],
    ) -> int:
        projected = pos

        buy_cap = max(0, HARD_POS_LIMIT - projected)
        if buy_cap > 0:
            for ask_px, ask_vol in sorted(od.sell_orders.items()):
                if ask_px > fair - TAKE_EDGE or buy_cap <= 0:
                    break
                qty = min(BASE_ORDER_QTY, -ask_vol, buy_cap)
                if qty > 0:
                    orders.append(Order(product, ask_px, qty))
                    projected += qty
                    buy_cap -= qty

        sell_cap = max(0, HARD_POS_LIMIT + projected)
        if sell_cap > 0:
            for bid_px, bid_vol in sorted(od.buy_orders.items(), reverse=True):
                if bid_px < fair + TAKE_EDGE or sell_cap <= 0:
                    break
                qty = min(BASE_ORDER_QTY, bid_vol, sell_cap)
                if qty > 0:
                    orders.append(Order(product, bid_px, -qty))
                    projected -= qty
                    sell_cap -= qty

        return projected

    @staticmethod
    def _make_quotes(
        product: str,
        fair: float,
        edge: int,
        pos: int,
        od: OrderDepth,
        orders: List[Order],
    ) -> None:
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        reservation = fair - INV_SKEW_PER_UNIT * pos
        bid_px = int(math.floor(reservation - edge))
        ask_px = int(math.ceil(reservation + edge))

        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            return

        buy_room = SOFT_POS_LIMIT - pos
        sell_room = SOFT_POS_LIMIT + pos

        buy_qty = min(BASE_ORDER_QTY, max(0, buy_room))
        sell_qty = min(BASE_ORDER_QTY, max(0, sell_room))

        if pos >= SOFT_POS_LIMIT:
            buy_qty = 0
        if pos <= -SOFT_POS_LIMIT:
            sell_qty = 0

        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))

    def run(self, state: TradingState):
        data = self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {p: [] for p in PRODUCTS}

        for product in PRODUCTS:
            od = state.order_depths.get(product)
            snap = self._book_snapshot(od)
            if snap is None or od is None:
                continue

            pos = state.position.get(product, 0)
            fair = self._fair_value(product, snap["weighted_mid"], data)
            edge = self._quote_edge(product, snap["spread"], data)

            projected_pos = self._take_mispriced_book(
                product, fair, pos, od, result[product]
            )
            self._make_quotes(product, fair, edge, projected_pos, od, result[product])

        return result, 0, self._save_state(data)
