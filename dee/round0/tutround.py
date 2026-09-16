from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    def __init__(self):
        self.limit = 80
        self.emerald_fair = 10000

    def run(self, state: TradingState):
        result = {}
        for product in ['EMERALDS', 'TOMATOES']:
            if product not in state.order_depths: continue
                
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)
            
            if product == 'EMERALDS':
                base_bid = self.emerald_fair - 2
                base_ask = self.emerald_fair + 2
                skew = 2 * (pos / self.limit)
                dynamic_bid = round(base_bid - skew)
                dynamic_ask = round(base_ask - skew)
                
                if dynamic_bid >= dynamic_ask:
                    dynamic_bid = dynamic_ask - 1

                if pos < self.limit:
                    orders.append(Order(product, dynamic_bid, self.limit - pos))
                if pos > -self.limit:
                    orders.append(Order(product, dynamic_ask, -self.limit - pos))

            elif product == 'TOMATOES':
                bids = order_depth.buy_orders
                asks = order_depth.sell_orders
                
                if bids and asks:
                    best_bid = max(bids.keys())
                    best_ask = min(asks.keys())
                    
                    if pos < self.limit:
                        buy_price = best_bid + 1
                        if buy_price < best_ask:
                            orders.append(Order(product, buy_price, self.limit - pos))
                            
                    if pos > -self.limit:
                        sell_price = best_ask - 1
                        if sell_price > best_bid:
                            orders.append(Order(product, sell_price, -self.limit - pos))

            result[product] = orders
            
        return result, 0, ""