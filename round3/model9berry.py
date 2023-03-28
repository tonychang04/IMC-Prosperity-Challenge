import json
import math
from typing import Any
from typing import Dict, List

import numpy as np

from datamodel import Order, ProsperityEncoder, Symbol, TradingState
from datamodel import OrderDepth


# only needed when using visualizer
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""


logger = Logger()


class Trader:

    def __init__(self):
        self.positions = {'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES':0}  # initialize position dictionary
        self.prices = {"BANANAS": [], "COCONUTS": [], "PINA_COLADAS": [], "BERRIES":[]}
        self.spread = []  # used to calculate the spread between coconuts and pina coladas
        self.position_limit = {"COCONUTS": 600, "PINA_COLADAS": 300, "BANANAS": 20, "BERRIES": 250}
        self.acc_flag = 0  # for berries

    # self.temp = []

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # print("self.positions[berries] of BANANAS: " + str(self.positions['BANANAS']))

        verbose_pair_trading = True # change this to True to enable printing logs
        verbose_banana_trading = True # change this to True to enable printing logs
        visualizer = True # change this to True to upload log to visualizer

        # helper functions
        def get_best_bid(order_depth: OrderDepth) -> int:
            return max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None

        def get_best_ask(order_depth: OrderDepth) -> int:
            return min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None

        def get_market_price(best_bid, best_ask) -> int:
            return (best_bid + best_ask) / 2 if best_bid and best_ask else None

        def get_z_score(price, mean, std):
            return (price - mean) / std


        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            if product == 'BERRIES':

                berries = 'BERRIES'

                order_depth: OrderDepth = state.order_depths[berries]

                best_bid = get_best_bid(order_depth)
                best_ask = get_best_ask(order_depth)

                today_price = get_market_price(best_bid, best_ask)
                if today_price is not None: self.prices[berries].append(today_price)

                cur_time = state.timestamp
                
                # 可以赌低开
                if cur_time < 200000:
                    # 尽量买到30
                    if today_price < 3850:
                        first_aim = self.position_limit[berries] - self.positions[berries]
                        buy_volume = min(abs(first_aim), abs(order_depth.sell_orders[best_ask]))
                        print("BUY berries", str(buy_volume) + "x", best_ask)
                        orders = [Order(berries, best_ask, buy_volume)]
                        self.positions[berries] += buy_volume
                        result[berries] = orders
                    elif cur_time > 100000: 
                        SMA_10 = np.mean(self.prices[berries][-10:])
                        if today_price < SMA_10:
                            first_aim = self.position_limit[berries] - self.positions[berries]
                            buy_volume = min(abs(first_aim), abs(order_depth.sell_orders[best_ask]))
                            print("BUY berries", str(buy_volume) + "x", best_ask)
                            orders = [Order(berries, best_ask, buy_volume)]
                            self.positions[berries] += buy_volume
                            result[berries] = orders

                elif cur_time >= 200000 and cur_time < 400000:
                    SMA_20 = np.mean(self.prices[berries][-20:])
                    
                    if today_price < SMA_20:
                        # 买多点
                        third_aim = self.position_limit[berries] - self.positions[berries]
                        buy_volume = min(abs(third_aim), abs(order_depth.sell_orders[best_ask]))
                    else:
                        # 买少点
                        buy_volume = min(10,abs(order_depth.sell_orders[best_ask]))

                    print("BUY berries", str(buy_volume) + "x", best_ask)
                    orders = [Order(berries, best_ask, buy_volume)]
                    self.positions[berries] += buy_volume
                    result[berries] = orders
                # 400k - 470k flag > 0.65 才卖
                elif cur_time >= 430000 and cur_time <= 480000:
                    st_mean = np.mean(self.prices[berries][3900:])
                    st_std = np.std(self.prices[berries][3900:])
                    flag = (today_price - st_mean) / st_std
                    if flag > 0.9:
                        max_pos = self.position_limit[berries] + self.positions[berries]
                        sell_volume = min(max_pos, abs(order_depth.buy_orders[best_bid]))
                        print("SELL berries", str(sell_volume) + "x", best_bid)
                        orders = [Order(product, best_bid, -sell_volume)]
                        self.positions[product] -= sell_volume
                        result[product] = orders

                # 470k 之后疯狂做空-250
                elif cur_time > 480000 and self.positions[berries] > -250:
                    max_pos = self.position_limit[berries] + self.positions[berries]
                    sell_volume = min(max_pos, abs(order_depth.buy_orders[best_bid]))
                    print("SELL berries", str(sell_volume) + "x", best_bid)
                    orders = [Order(product, best_bid, -sell_volume)]
                    self.positions[product] -= sell_volume
                    result[product] = orders

        if visualizer:
            logger.flush(state, result)
        return result
