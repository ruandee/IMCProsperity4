import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    LIMIT_UNDERLYING = 200
    LIMIT_OPTIONS = 200
    TRADE_EDGE = 2.0
    BASE_SIGMA = 1794.66

    def get_lo_wang_adjustment(self) -> float:
        return 1.203149

    def norm_cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def bs_price_and_delta(self, S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
        if T <= 0:
            return max(0.0, S - K), 1.0 if S > K else 0.0

        d1 = (math.log(S / K) + 0.5 * (sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        theo_price = S * self.norm_cdf(d1) - K * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)

        return theo_price, delta

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        T_YEARS = 4.0 / 252.0
        underlying = "VELVETFRUIT_EXTRACT"
        vouchers = [
            "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
            "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"
        ]

        if underlying not in state.order_depths:
            return result, conversions, trader_data

        und_order_depth = state.order_depths[underlying]
        if not und_order_depth.sell_orders or not und_order_depth.buy_orders:
            return result, conversions, trader_data

        best_ask = min(und_order_depth.sell_orders.keys())
        best_bid = max(und_order_depth.buy_orders.keys())
        und_mid = (best_ask + best_bid) / 2.0

        lw_factor = self.get_lo_wang_adjustment()
        adjusted_sigma = self.BASE_SIGMA * lw_factor

        total_delta = 0.0

        for opt in vouchers:
            if opt not in state.order_depths:
                continue

            strike = float(opt.split('_')[1])
            theo_price, opt_delta = self.bs_price_and_delta(und_mid, strike, T_YEARS, adjusted_sigma)

            opt_depth = state.order_depths[opt]
            opt_pos = state.position.get(opt, 0)
            orders: List[Order] = []
            trade_qty = 0

            for ask_price, ask_vol in list(opt_depth.sell_orders.items()):
                if ask_price < theo_price - self.TRADE_EDGE:
                    buy_qty = min(-ask_vol, self.LIMIT_OPTIONS - opt_pos - trade_qty)
                    if buy_qty > 0:
                        orders.append(Order(opt, ask_price, buy_qty))
                        trade_qty += buy_qty

            for bid_price, bid_vol in list(opt_depth.buy_orders.items()):
                if bid_price > theo_price + self.TRADE_EDGE:
                    sell_qty = min(bid_vol, self.LIMIT_OPTIONS + opt_pos + trade_qty)
                    if sell_qty > 0:
                        orders.append(Order(opt, bid_price, -sell_qty))
                        trade_qty -= sell_qty

            if len(orders) > 0:
                result[opt] = orders

            total_delta += (opt_pos + trade_qty) * opt_delta

        und_pos = state.position.get(underlying, 0)
        target_und_pos = -total_delta
        qty_to_trade = int(round(target_und_pos - und_pos))

        und_orders = []
        if qty_to_trade > 0:
            und_orders.append(Order(underlying, int(best_ask + 5), qty_to_trade))
        elif qty_to_trade < 0:
            und_orders.append(Order(underlying, int(best_bid - 5), qty_to_trade))

        if und_orders:
            result[underlying] = und_orders

        return result, conversions, trader_data