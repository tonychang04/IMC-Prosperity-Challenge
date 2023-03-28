from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np

class Trader:

    def __init__(self):
        self.dolphins = []
        self.dolphins_flag = 0
        self.position = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order depths
        def get_best_bid(order_depth: OrderDepth) -> int:
            return max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None

        def get_best_ask(order_depth: OrderDepth) -> int:
            return min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None

        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product == 'DIVING_GEAR':
                self.dolphins.append(state.observations['DOLPHIN_SIGHTINGS'])
                if len(self.dolphins) < 50: continue

                order_depth: OrderDepth = state.order_depths[product]

                # Get the current best bid and ask prices
                best_bid = get_best_bid(order_depth)
                best_ask = get_best_ask(order_depth)

                if not best_bid or not best_ask: continue

                position_limit = 50
                net_position = self.position

                if np.mean(self.dolphins[-20:]) > np.mean(self.dolphins[-50]) and (self.dolphins_flag == 0 or self.dolphins_flag == 1):

                    max_pos = position_limit - net_position

                    buy_volume = min(max_pos, abs(order_depth.sell_orders[best_ask]))
                    print("BUY", str(buy_volume) + "x", best_ask)
                    orders = [Order(product, best_ask, buy_volume)]
                    self.position += buy_volume
                    result[product] = orders
                    self.dolphins_flag = -1
                elif np.mean(self.dolphins[-20:]) < np.mean(self.dolphins[-50]) and (self.dolphins_flag == 0 or self.dolphins_flag == -1):

                    max_pos = position_limit + net_position
                    sell_volume = min(max_pos, abs(order_depth.buy_orders[best_bid]))
                    print("SELL", str(sell_volume) + "x", best_bid)
                    orders = [Order(product, best_bid, -sell_volume)]
                    self.position -= sell_volume
                    result[product] = orders
                    self.dolphins_flag = 1








        return result