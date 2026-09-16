from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""
        product = "HYDROGEL_PACK"
        
        if product in state.order_depths:
            current_position = state.position.get(product, 0)
            if current_position == 0:
                order_depth: OrderDepth = state.order_depths[product]
                if len(order_depth.sell_orders) > 0:
                    best_ask = list(order_depth.sell_orders.keys())[0]
                    print(f"ITERATION {state.timestamp}: Ordered {product} at {best_ask}.")
                    result[product] = [Order(product, best_ask, 1)]
        return result, conversions, traderData