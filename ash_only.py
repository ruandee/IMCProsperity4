import math
import json
from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    def __init__(self):
        self.position_limit = 80
        self.risk_factor = 0.075
        self.ou_const = 5.35
        self.ema_alpha = 0.6 # fair 0.2152
        self.current_ema = None

        self.layers = [
            (0.4 * self.ou_const, 15),
            (0.8 * self.ou_const, 25),
            (1.5 * self.ou_const, 40)
        ]

    def run(self, state: TradingState):
        executed_volume = 0
        for product, trades in state.own_trades.items():
            for trade in trades:
                executed_volume += abs(trade.quantity)

        result = {}
        product = "ASH_COATED_OSMIUM"

        if state.traderData:
            try:
                data = json.loads(state.traderData)
                self.current_ema = data.get("ema")
            except:
                self.current_ema = None

        if product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                return result, 0, json.dumps({"ema": self.current_ema})

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            if self.current_ema is None:
                self.current_ema = mid_price
            else:
                self.current_ema = (self.ema_alpha * mid_price) + (1 - self.ema_alpha) * self.current_ema

            res_price = self.current_ema - (pos * self.risk_factor)
            
            orders: List[Order] = []
            buy_cap = self.position_limit - pos
            sell_cap = -self.position_limit - pos

            remaining_sell = sell_cap
            for offset, layer_qty in self.layers:
                if remaining_sell < 0:
                    price = math.ceil(res_price + offset)
                    price = max(price, best_bid + 1)
                    
                    vol = max(-layer_qty, remaining_sell)
                    orders.append(Order(product, int(price), int(vol)))
                    remaining_sell -= vol

            remaining_buy = buy_cap
            for offset, layer_qty in self.layers:
                if remaining_buy > 0:
                    price = math.floor(res_price - offset)
                    price = min(price, best_ask - 1)
                    
                    vol = min(layer_qty, remaining_buy)
                    orders.append(Order(product, int(price), int(vol)))
                    remaining_buy -= vol

            result[product] = orders
        
        ordered_volume = 0
        for prod, orders_list in result.items():
            for order in orders_list:
                ordered_volume += abs(order.quantity)
        
        print(f"Executed: {executed_volume} Ordered: {ordered_volume}")
        
        traderData = json.dumps({"ema": self.current_ema})
        return result, 0, traderData