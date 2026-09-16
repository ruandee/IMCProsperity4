import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    THETA = 5250.71
    KAPPA = 6254.50
    SIGMA = 1794.66
    R = 0.0
    
    T_YEARS = 5.0 / 252.0 
    
    TRADE_EDGE = 4 

    def norm_cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def get_ou_variance(self, sigma: float, kappa: float, T: float) -> float:
        if kappa == 0:
            return (sigma**2) * T
        return (sigma**2 / (2 * kappa)) * (1 - math.exp(-2 * kappa * T))

    def get_ou_forward_price(self, S: float, theta: float, kappa: float, T: float) -> float:
        if kappa == 0:
            return S
        return S * math.exp(-kappa * T) + theta * (1 - math.exp(-kappa * T))

    def ou_option_price(self, S: float, K: float, T: float, r: float, 
                        sigma: float, kappa: float, theta: float, 
                        option_type: str = 'call') -> float:
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        F = self.get_ou_forward_price(S, theta, kappa, T)
        ou_var = self.get_ou_variance(sigma, kappa, T)
        
        if ou_var <= 0:
            if option_type == 'call':
                return max(F - K, 0)
            else:
                return max(K - F, 0)

        ou_vol = math.sqrt(ou_var)
        d1 = (math.log(F / K) + 0.5 * ou_var) / ou_vol
        d2 = d1 - ou_vol
        df = math.exp(-r * T)

        if option_type == 'call':
            price = df * (F * self.norm_cdf(d1) - K * self.norm_cdf(d2))
        else:
            price = df * (K * self.norm_cdf(-d2) - F * self.norm_cdf(-d1))

        return price

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""
        
        spot_price = self.THETA 
        underlying = "VELVETFRUIT_EXTRACT"
        
        if underlying in state.order_depths:
            ob = state.order_depths[underlying]
            if len(ob.sell_orders) > 0 and len(ob.buy_orders) > 0:
                best_ask = min(ob.sell_orders.keys())
                best_bid = max(ob.buy_orders.keys())
                spot_price = (best_ask + best_bid) / 2.0

        ## VOucher trading
        for product in state.order_depths.keys():
            if product.startswith("VEV_"):
                strike = float(product.split("_")[1])
                
                option_type = 'call' 
                
                fair_value = self.ou_option_price(
                    S=spot_price, K=strike, T=self.T_YEARS, r=self.R, 
                    sigma=self.SIGMA, kappa=self.KAPPA, theta=self.THETA, 
                    option_type=option_type
                )
                
                ob = state.order_depths[product]
                orders: List[Order] = []
                
                if len(ob.buy_orders) > 0:
                    best_bid = max(ob.buy_orders.keys())
                    if best_bid > fair_value + self.TRADE_EDGE:
                        bid_vol = ob.buy_orders[best_bid] 
                        orders.append(Order(product, best_bid, -bid_vol))
            
                if len(ob.sell_orders) > 0:
                    best_ask = min(ob.sell_orders.keys())
                    if best_ask < fair_value - self.TRADE_EDGE:
                        ask_vol = abs(ob.sell_orders[best_ask]) 
                        orders.append(Order(product, best_ask, ask_vol))
                
                if len(orders) > 0:
                    result[product] = orders

        return result, conversions, traderData