import json
from typing import Any
from typing import Dict, List

import numpy as np

from datamodel import Order, ProsperityEncoder, Symbol, TradingState
from datamodel import OrderDepth

def __init__(self):
    self.positions = {'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0,
                      'DIVING_GEAR': 0, "BERRIES": 0, "DIP": 0,
                      "UKULELE": 0, "PICNIC_BASKET": 0, "BAGUETTE": 0}  # initialize position dictionary

    self.prices = {"BANANAS": [], "COCONUTS": [], "PINA_COLADAS": [], "DOLPHINS": [], "DIVING_GEAR": [],
                   "BERRIES": [], "BAGUETTE": [], "DIP": [], "UKULELE": [], "COMBINE": [], "PICNIC_BASKET": []}
    self.spread = []  # used to calculate the spread between coconuts and pina coladas
    self.beta = []
    self.zscore = []
    self.costs = [[], []]  # cocunuts prices, pina coladas prices
    self.position_limit = {"COCONUTS": 600, "PINA_COLADAS": 300, "BANANAS": 20, "DIVING_GEAR": 50, "BERRIES": 250,
                           "BAGUETTE": 150, "DIP": 300,
                           "UKULELE": 70, "PICNIC_BASKET": 70}
    self.dolphins_flag = 0
    self.bananas_flag = 0

    self.diff = []  # diff between combine and basket

    # self.temp = []


def run(self, state: TradingState) -> Dict[str, List[Order]]:
    """
    Only method required. It takes all buy and sell orders for all symbols as an input,
    and outputs a list of orders to be sent
    """
    # Initialize the method output dict as an empty dict
    result = {}

    # print("net_position of BANANAS: " + str(self.positions['BANANAS']))

    verbose_pair_trading = False  # change this to True to enable printing logs
    verbose_banana_trading = False  # change this to True to enable printing logs
    visualizer = True  # change this to True to upload log to visualizer

    # helper functions
    def get_best_bid(order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None

    def get_best_ask(order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None

    def get_market_price(best_bid, best_ask) -> int:
        return (best_bid + best_ask) / 2 if best_bid and best_ask else None

    def get_z_score(price, mean, std):
        return (price - mean) / std

    def buy(product, max_pos, order_depth, best_ask):
        buy_volume = min(max_pos, abs(order_depth.sell_orders[best_ask]))
        print(f'BUY {product}', str(buy_volume) + "x", best_ask)
        orders = [Order(product, best_ask, buy_volume)]
        self.positions[product] += buy_volume
        return orders

    def sell(product, max_pos, order_depth, best_bid):
        sell_volume = min(max_pos, abs(order_depth.buy_orders[best_bid]))
        print(f'SELL {product}', str(sell_volume) + "x", best_bid)
        orders = [Order(product, best_bid, -sell_volume)]
        self.positions[product] -= sell_volume
        return orders

    def close_position(product, order_depth, best_ask, best_bid):
        max_pos = abs(self.positions[product])
        if self.positions[product] < 0:
            result[product] = buy(product, max_pos, order_depth, best_ask)
        else:
            result[product] = sell(product, max_pos, order_depth, best_bid)

    # Iterate over all the keys (the available products) contained in the order depths
    for product in state.order_depths.keys():

        if product == "PICNIC_BASKET":

            bag, dip, uku, combine = 'BAGUETTE', 'DIP', 'UKULELE', 'COMBINE'

            order_depth_basket: OrderDepth = state.order_depths[product]
            order_depth_bag: OrderDepth = state.order_depths[bag]
            order_depth_dip: OrderDepth = state.order_depths[dip]
            order_depth_uku: OrderDepth = state.order_depths[uku]

            best_bid_basket = get_best_bid(order_depth_basket)
            best_ask_basket = get_best_ask(order_depth_basket)
            best_bid_bag = get_best_bid(order_depth_bag)
            best_ask_bag = get_best_ask(order_depth_bag)
            best_bid_dip = get_best_bid(order_depth_dip)
            best_ask_dip = get_best_ask(order_depth_dip)
            best_bid_uku = get_best_bid(order_depth_uku)
            best_ask_uku = get_best_ask(order_depth_uku)

            today_price_basket = get_market_price(best_bid_basket, best_ask_basket)
            today_price_bag = get_market_price(best_bid_bag, best_ask_bag)
            today_price_dip = get_market_price(best_bid_dip, best_ask_dip)
            today_price_uku = get_market_price(best_bid_uku, best_ask_uku)

            if today_price_basket is not None: self.prices[product].append(today_price_basket)
            if today_price_bag is not None: self.prices[bag].append(today_price_bag)
            if today_price_dip is not None: self.prices[dip].append(today_price_dip)
            if today_price_uku is not None: self.prices[uku].append(today_price_uku)

            if today_price_uku == None or today_price_bag == None or today_price_dip == None or today_price_basket == None:
                continue
            today_price_combine = 2 * today_price_bag + 4 * today_price_dip + today_price_uku
            best_bid_combine = 2 * best_bid_bag + 4 * best_bid_dip + best_bid_uku
            best_ask_combine = 2 * best_ask_bag + 4 * best_ask_dip + best_ask_uku

            # print('-----------------------')
            # print(best_bid_combine,best_ask_basket)
            # print(best_ask_combine,best_bid_basket)
            # print('-----------------------')
            today_diff = today_price_basket - today_price_combine
            self.diff.append(today_diff)

            if len(self.diff) <= 20: continue
            #diff_mean = np.mean(self.diff[-20:])
            #diff_std = np.std(self.diff[-20:])
            #flag = get_z_score(today_diff, diff_mean, diff_std)

            self.zscore.append(flag)

            if flag < -1.68:
                # sell combine buy basket\
                max_pos_basket = self.position_limit[product] - self.positions[product]
                buy_volume_basket = min(max_pos_basket, abs(order_depth_basket.sell_orders[best_ask_basket]))
                print("BUY basket", str(buy_volume_basket) + "x", best_ask_basket)
                orders = [Order(product, best_ask_basket, buy_volume_basket)]
                self.positions[product] += buy_volume_basket
                result[product] = orders

                max_pos_bag = min(self.position_limit[bag] + self.positions[bag], 2 * buy_volume_basket)
                # max_pos_bag = self.position_limit[bag] + self.positions[bag]
                sell_volume = min(max_pos_bag, abs(order_depth_bag.buy_orders[best_bid_bag]))
                print("SELL bag", str(sell_volume) + "x", best_bid_bag)
                orders = [Order(bag, best_bid_bag, -sell_volume)]
                self.positions[bag] -= sell_volume
                result[bag] = orders

                max_pos_dip = min(self.position_limit[dip] + self.positions[dip], 4 * buy_volume_basket)
                # max_pos_dip = self.position_limit[dip] + self.positions[dip]
                sell_volume = min(max_pos_dip, abs(order_depth_dip.buy_orders[best_bid_dip]))
                print("SELL dip", str(sell_volume) + "x", best_bid_dip)
                orders = [Order(dip, best_bid_dip, -sell_volume)]
                self.positions[dip] -= sell_volume
                result[dip] = orders

                max_pos_uku = min(self.position_limit[uku] + self.positions[uku], buy_volume_basket)
                # max_pos_uku = self.position_limit[uku] + self.positions[uku]
                sell_volume = min(max_pos_uku, abs(order_depth_uku.buy_orders[best_bid_uku]))
                print("SELL uku", str(sell_volume) + "x", best_bid_uku)
                orders = [Order(uku, best_bid_uku, -sell_volume)]
                self.positions[uku] -= sell_volume
                result[uku] = orders

            elif flag > 1.68:
                # buy combine sell basket
                max_pos_basket = self.position_limit[product] + self.positions[product]
                sell_volume_basket = min(max_pos_basket, abs(order_depth_basket.buy_orders[best_bid_basket]))
                print("SELL basket", str(sell_volume_basket) + "x", best_bid_basket)
                orders = [Order(product, best_bid_basket, -sell_volume_basket)]
                self.positions[product] -= sell_volume_basket
                result[product] = orders

                max_pos_bag = min(self.position_limit[bag] - self.positions[bag], 2 * sell_volume_basket)
                # max_pos_bag = self.position_limit[bag] - self.positions[bag]
                buy_volume = min(max_pos_bag, abs(order_depth_bag.sell_orders[best_ask_bag]))
                print("BUY bag", str(buy_volume) + "x", best_ask_bag)
                orders = [Order(bag, best_ask_bag, buy_volume)]
                self.positions[bag] += buy_volume
                result[bag] = orders

                max_pos_dip = min(self.position_limit[dip] - self.positions[dip], 4 * sell_volume_basket)
                # max_pos_dip = self.position_limit[dip] - self.positions[dip]
                buy_volume = min(max_pos_dip, abs(order_depth_dip.sell_orders[best_ask_dip]))
                print("BUY dip", str(buy_volume) + "x", best_ask_dip)
                orders = [Order(dip, best_ask_dip, buy_volume)]
                self.positions[dip] += buy_volume
                result[dip] = orders

                max_pos_uku = min(self.position_limit[uku] - self.positions[uku], sell_volume_basket)
                # max_pos_uku = self.position_limit[uku] - self.positions[uku]
                buy_volume = min(max_pos_uku, abs(order_depth_uku.sell_orders[best_ask_uku]))
                print("BUY uku", str(buy_volume) + "x", best_ask_uku)
                orders = [Order(uku, best_ask_uku, buy_volume)]
                self.positions[uku] += buy_volume
                result[uku] = orders

    return result