import json
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
            '''
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
            '''




            if product == "BERRIES":
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
                    if today_price < 3825:
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
                        buy_volume = min(10, abs(order_depth.sell_orders[best_ask]))

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

            elif product == "DIVING_GEAR":
                self.prices["DOLPHINS"].append(state.observations['DOLPHIN_SIGHTINGS'])
                if len(self.prices["DOLPHINS"]) < 50: continue

                order_depth: OrderDepth = state.order_depths[product]

                # Get the current best bid and ask prices
                best_bid = get_best_bid(order_depth)
                best_ask = get_best_ask(order_depth)

                price = get_market_price(best_bid, best_ask)

                if price:
                    self.prices[product].append(price)

                if not best_bid or not best_ask: continue

                position_limit = 50
                net_position_dg = self.positions["DIVING_GEAR"]
                orders = []
                if np.mean(self.prices["DOLPHINS"][-20:]) > np.mean(self.prices["DOLPHINS"][-50]) and (
                        self.dolphins_flag == 0 or self.dolphins_flag == 1):

                    max_pos = position_limit - net_position_dg

                    buy_volume = min(max_pos, abs(order_depth.sell_orders[best_ask]))
                    print("BUY", str(buy_volume) + "x", best_ask)
                    orders.append(Order(product, best_ask, buy_volume))
                    self.positions["DIVING_GEAR"] += buy_volume
                    self.dolphins_flag = -1

                elif np.mean(self.prices["DOLPHINS"][-20:]) < np.mean(self.prices["DOLPHINS"][-50]) and (
                        self.dolphins_flag == 0 or self.dolphins_flag == -1):

                    max_pos = position_limit + net_position_dg
                    sell_volume = min(max_pos, abs(order_depth.buy_orders[best_bid]))
                    print("SELL", str(sell_volume) + "x", best_bid)
                    orders.append(Order(product, best_bid, -sell_volume))
                    self.positions["DIVING_GEAR"] -= sell_volume
                    self.dolphins_flag = 1


                elif self.positions["DIVING_GEAR"] == -50 and len(
                        self.prices[product]) > 10 and price and price < np.mean(
                    self.prices[product][-10:]):
                    # buy co
                    buy_volume_d = min(abs(net_position_dg),
                                       abs(order_depth.sell_orders[best_ask]))
                    orders.append((Order(product, best_ask, buy_volume_d)))
                    self.positions[product] += buy_volume_d

                elif self.positions["DIVING_GEAR"] == 50 and len(
                        self.prices["DIVING_GEAR"]) > 10 and price and price > np.mean(
                    self.prices[product][-10:]):
                    #  sell
                    sell_volume_d = min(abs(net_position_dg),
                                        abs(order_depth.buy_orders[best_bid]))
                    orders.append(Order(product, best_bid, sell_volume_d))
                    self.positions[product] -= sell_volume_d

                if len(orders) > 0:
                    result[product] = orders

            elif product == 'COCONUTS':

                pina = 'PINA_COLADAS'
                coconut = 'COCONUTS'

                # Retrieve the Order Depth containing all the market BUY and SELL orders for BANANAS
                order_depth_co: OrderDepth = state.order_depths[coconut]
                order_depth_pi: OrderDepth = state.order_depths[pina]

                # Get the current best bid and ask prices
                best_bid_co = get_best_bid(order_depth_co)
                best_ask_co = get_best_ask(order_depth_co)

                # Calculate the mid price(market price) as the average of the best bid and ask prices
                today_price_co = get_market_price(best_bid_co, best_ask_co)
                if today_price_co is not None: self.prices["COCONUTS"].append(today_price_co)

                best_bid_pina = get_best_bid(order_depth_pi)
                best_ask_pina = get_best_ask(order_depth_pi)

                today_price_pina = get_market_price(best_bid_pina, best_ask_pina)
                if today_price_pina is not None: self.prices["PINA_COLADAS"].append(today_price_pina)

                if not today_price_pina or not today_price_co: continue

                self.costs[0].append(today_price_co)
                self.costs[1].append(today_price_pina)

                window = 30
                if len(self.costs[0]) and len(self.costs[1]) < window:
                    beta = np.linalg.lstsq(np.vstack([self.costs[1], np.ones(len(self.costs[1]))]).T,
                                           np.array(self.costs[0]),
                                           rcond=None)[0][0]
                else:
                    beta = np.linalg.lstsq(np.vstack([self.costs[1][-window:], np.ones(window)]).T,
                                           np.array(self.costs[0][-window:]),
                                           rcond=None)[0][0]

                spread = today_price_co - today_price_pina * beta
                # self.temp.append([math.log(today_price_co), math.log(today_price_pina)]) # only used to get n

                self.spread.append(spread)
                self.beta.append(beta)
                if len(self.spread) < window: continue

                spread_mean = np.mean(np.array(self.costs[0][-window:]) - np.array(self.costs[1][-window:]) * np.array(
                    self.beta[-window:]))
                spread_std = np.std(np.array(self.costs[0][-window:]) - np.array(self.costs[1][-window:]) * np.array(
                    self.beta[-window:]))
                zscore = get_z_score(spread, spread_mean, spread_std)

                self.zscore.append(zscore)

                net_position_co = self.positions[coconut]
                net_position_pi = self.positions[pina]
                position_limit_co = self.position_limit[coconut]
                position_limit_pi = self.position_limit[pina]

                if verbose_pair_trading:
                    print("------PAIR--------")
                    print("today_price_co: " + str(today_price_co))
                    print("today_price_pina: " + str(today_price_pina))
                    print("best_bid_co: " + str(best_bid_co))
                    print("best_ask_co: " + str(best_ask_co))
                    print("net_position of COCONUTS: " + str(net_position_co))
                    print("best_bid_pina: " + str(best_bid_pina))
                    print("best_ask_pina: " + str(best_ask_pina))
                    print("net_position of PINA_COLADAS: " + str(net_position_pi))
                    print("spread: " + str(spread))
                    print("---------------------")

                print("Cocunut zscore: " + str(zscore))

                if zscore >= 2:
                    # sell co buy pi
                    max_pos_co = position_limit_co + net_position_co
                    max_pos_pi = position_limit_pi - net_position_pi

                    # calculate the volume to sell and buy
                    sell_volume_co = min(max_pos_co,
                                         abs(order_depth_co.buy_orders[best_bid_co]))
                    buy_volume_pi = min(max_pos_pi,
                                        abs(order_depth_pi.sell_orders[best_ask_pina]))

                    print("SELL coconut", str(sell_volume_co) + "x", best_bid_co)
                    print("BUY pina", str(buy_volume_pi) + "x", best_ask_pina)

                    # print("BUY pina", str(buy_volume_pi) + "x", best_ask_pina)

                    # send order update position
                    orders_co = [Order(coconut, best_bid_co, -sell_volume_co)]
                    orders_pi = [Order(pina, best_ask_pina, buy_volume_pi)]
                    self.positions[coconut] -= sell_volume_co
                    self.positions[pina] += buy_volume_pi
                    #  self.positions[pina] += buy_volume_pi
                    result[coconut] = orders_co
                    result[pina] = orders_pi

                elif zscore <= -2:
                    # buy co sell pi
                    max_pos_pi = position_limit_pi + net_position_pi
                    max_pos_co = position_limit_co - net_position_co

                    #  sell_volume_pi = min(max_pos_pi,
                    #                       abs(order_depth_pi.buy_orders[best_bid_pina]))

                    sell_volume_pi = min(max_pos_pi,
                                         abs(order_depth_pi.buy_orders[best_bid_pina]))

                    buy_volume_co = min(max_pos_co,
                                        abs(order_depth_co.sell_orders[best_ask_co]))

                    #  print("SELL pina", str(sell_volume_pi) + "x", best_bid_pina)
                    print("BUY co", str(buy_volume_co) + "x", best_ask_co)
                    print("SELL pina", str(sell_volume_pi) + "x", best_bid_pina)

                    # send order update position
                    orders_co = [Order(coconut, best_ask_co, buy_volume_co)]
                    orders_pi = [Order(pina, best_bid_pina, -sell_volume_pi)]
                    # orders_pi = [Order(pina, best_bid_pina, -sell_volume_pi)]
                    self.positions[coconut] += buy_volume_co
                    self.positions[pina] -= sell_volume_pi
                    result[coconut] = orders_co
                    result[pina] = orders_pi
                # result[pina] = orders_pi
                '''
                elif abs(zscore) < 0.5:

                    if net_position_co > 500 and len(self.prices[coconut]) > 10 and today_price_co > np.mean(
                            self.prices[coconut][-10:]) and best_bid_co:
                        # sell co
                        sell_volume_co = min(abs(net_position_co),
                                             abs(order_depth_co.buy_orders[best_bid_co]))

                        orders_co = [Order(coconut, best_bid_co, -sell_volume_co)]

                        print("SELL coconut", str(sell_volume_co) + "x", best_bid_co)
                        self.positions[coconut] -= sell_volume_co
                        result[coconut] = orders_co

                    if net_position_co < -500 and len(self.prices[coconut]) > 10 and today_price_co < np.mean(
                            self.prices[coconut][-10:]):
                        # buy co
                        buy_volume_co = min(abs(net_position_co),
                                            abs(order_depth_co.sell_orders[best_ask_co]))
                        print("BUY coconut", str(buy_volume_co) + "x", best_ask_co)
                        orders_co = [Order(coconut, best_ask_co, buy_volume_co)]
                        self.positions[coconut] += buy_volume_co
                        result[coconut] = orders_co

                    if net_position_pi > 270 and len(self.prices[pina]) > 10 and today_price_pina > np.mean(
                            self.prices[pina][-10:]) and best_bid_pina:
                        # sell co
                        sell_volume_pi = min(abs(net_position_pi),
                                             abs(order_depth_pi.buy_orders[best_bid_pina]))

                        print("SELL pina", str(sell_volume_pi) + "x", best_bid_pina)
                        orders_pi = [Order(pina, best_bid_pina, -sell_volume_pi)]
                        self.positions[pina] -= sell_volume_pi
                        result[pina] = orders_pi

                    if net_position_pi < -270 and len(self.prices[pina]) > 10 and today_price_pina < np.mean(
                            self.prices[pina][-10:]):
                        # buy co
                        buy_volume_pi = min(abs(net_position_pi),
                                            abs(order_depth_pi.sell_orders[best_ask_pina]))

                        print("BUY pina", str(buy_volume_pi) + "x", best_ask_pina)
                        orders_pi = [Order(pina, best_ask_pina, buy_volume_pi)]
                        self.positions[pina] += buy_volume_pi
                        result[pina] = orders_pi
                '''




            elif product == "BANANAS":
                '''
                # Retrieve the Order Depth containing all the market BUY and SELL orders for BANANAS
                order_depth_ba: OrderDepth = state.order_depths[product]

                # Get the current best bid and ask prices
                best_bid = get_best_bid(order_depth_ba)
                best_ask = get_best_ask(order_depth_ba)

                # Calculate the mid price(market price) as the average of the best bid and ask prices
                today_price_ba = get_market_price(best_bid, best_ask)
                if today_price_ba is not None: self.prices[product].append(today_price_ba)



                if not best_bid or not best_ask: continue

                position_limit = self.position_limit[product]
                net_position_ba = self.positions[product]
                orders = []

                if len(self.prices[product]) < 50:
                    continue
                if np.mean(self.prices[product][-20:]) > np.mean(self.prices[product][-50]) and (
                        self.bananas_flag == 0 or self.bananas_flag == 1):

                    max_pos = position_limit - net_position_ba

                    buy_volume = min(max_pos, abs(order_depth_ba.sell_orders[best_ask]))
                    print("BUY", str(buy_volume) + "x", best_ask)
                    orders.append(Order(product, best_ask, buy_volume))
                    self.positions[product] += buy_volume
                    self.bananas_flag = -1

                elif np.mean(self.prices[product][-20:]) < np.mean(self.prices[product][-50]) and (
                        self.bananas_flag == 0 or self.bananas_flag == -1):

                    max_pos = position_limit + net_position_ba
                    sell_volume = min(max_pos, abs(order_depth_ba.buy_orders[best_bid]))
                    print("SELL", str(sell_volume) + "x", best_bid)
                    orders.append(Order(product, best_bid, -sell_volume))
                    self.positions[product] -= sell_volume
                    self.bananas_flag = 1


                elif self.positions[product] == -position_limit and len(
                        self.prices[product]) > 10 and today_price_ba and today_price_ba < np.mean(
                    self.prices[product][-10:]):
                    buy_volume_ba = min(abs(net_position_ba),
                                       abs(order_depth_ba.sell_orders[best_ask]))
                    orders.append((Order(product, best_ask, buy_volume_ba)))
                    self.positions[product] += buy_volume_ba

                elif self.positions[product] == position_limit and len(
                        self.prices[product]) > 10 and today_price_ba and today_price_ba > np.mean(
                    self.prices[product][-10:]):
                    #  sell
                    sell_volume_ba = min(abs(net_position_ba),
                                        abs(order_depth_ba.buy_orders[best_bid]))
                    orders.append(Order(product, best_bid, sell_volume_ba))
                    self.positions[product] -= sell_volume_ba

                if len(orders) > 0:
                    result[product] = orders
                '''
            elif product == "PEARLS":
                fair_price = 10000
                order_depth_pr: OrderDepth = state.order_depths["PEARLS"]
                orders = []
                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth_pr.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth_pr.sell_orders.keys())
                    best_ask_volume = order_depth_pr.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < fair_price:
                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY PEARL", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth_pr.buy_orders) != 0:
                    best_bid = max(order_depth_pr.buy_orders.keys())
                    best_bid_volume = order_depth_pr.buy_orders[best_bid]
                    if best_bid > fair_price:
                        print("SELL PEARL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above orders to the result dict
                result[product] = orders

            if state.timestamp == 99900:
                print(self.zscore)

        if visualizer:
            logger.flush(state, result)
        return result
