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
        self.positions = {'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0,
                          'DIVING_GEAR': 0, "BERRIES":0}  # initialize position dictionary
        self.prices = {"BANANAS": [], "COCONUTS": [], "PINA_COLADAS": [], "DOLPHINS": [], "DIVING_GEAR": [],"BERRIES":[],}
        self.spread = []  # used to calculate the spread between coconuts and pina coladas
        self.spread2 = []  # used to calculate the spread between pina coladas and coconuts
        self.position_limit = {"COCONUTS": 600, "PINA_COLADAS": 300, "BANANAS": 20, "DIVING_GEAR": 50, "BERRIES": 250}
        self.dolphins_flag = 0

        # self.temp = []

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # print("net_position of BANANAS: " + str(self.positions['BANANAS']))

        verbose_pair_trading = True  # change this to True to enable printing logs
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

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():
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

                spread = math.log(today_price_co) - math.log(today_price_pina)
                # self.temp.append([math.log(today_price_co), math.log(today_price_pina)]) # only used to get n

                self.spread.append(spread)
                spread_mean = np.mean(self.spread)
                spread_std = np.std(self.spread)
                zscore = get_z_score(spread, spread_mean, spread_std)

                net_position_co = self.positions[coconut]
                net_position_pi = self.positions[pina]
                position_limit_co = self.position_limit[coconut]
                position_limit_pi = self.position_limit[pina]

                if len(self.spread) < 10: continue

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

                if zscore >= 1.68:
                    # sell co sell pi
                    max_pos_co = position_limit_co + net_position_co
                    max_pos_pi = position_limit_pi + net_position_pi

                    # calculate the volume to sell and buy
                    sell_volume_co = min(max_pos_co,
                                         abs(order_depth_co.buy_orders[best_bid_co]))
                    sell_volume_pi = min(max_pos_pi,
                                         abs(order_depth_pi.buy_orders[best_bid_pina]))

                    print("SELL coconut", str(sell_volume_co) + "x", best_bid_co)
                    print("SELL pina", str(sell_volume_pi) + "x", best_bid_pina)

                    # print("BUY pina", str(buy_volume_pi) + "x", best_ask_pina)

                    # send order update position
                    orders_co = [Order(coconut, best_bid_co, -sell_volume_co)]
                    orders_pi = [Order(pina, best_bid_pina, -sell_volume_pi)]
                    self.positions[coconut] -= sell_volume_co
                    self.positions[pina] -= sell_volume_pi
                    #  self.positions[pina] += buy_volume_pi
                    result[coconut] = orders_co
                    result[pina] = orders_pi

                elif zscore <= -1.68:
                    # buy co buy pi
                    max_pos_pi = position_limit_pi - net_position_pi
                    max_pos_co = position_limit_co - net_position_co

                    #  sell_volume_pi = min(max_pos_pi,
                    #                       abs(order_depth_pi.buy_orders[best_bid_pina]))
                    buy_volume_co = min(max_pos_co,
                                        abs(order_depth_co.sell_orders[best_ask_co]))

                    buy_volume_pi = min(max_pos_pi,
                                        abs(order_depth_pi.sell_orders[best_ask_pina]))

                    #  print("SELL pina", str(sell_volume_pi) + "x", best_bid_pina)
                    print("BUY co", str(buy_volume_co) + "x", best_ask_co)
                    print("BUY pina", str(buy_volume_pi) + "x", best_ask_pina)

                    # send order update position
                    orders_co = [Order(coconut, best_ask_co, buy_volume_co)]
                    orders_pi = [Order(pina, best_ask_pina, buy_volume_pi)]
                    # orders_pi = [Order(pina, best_bid_pina, -sell_volume_pi)]
                    self.positions[coconut] += buy_volume_co
                    self.positions[pina] += buy_volume_pi
                    #  self.positions[pina] -= sell_volume_pi
                    result[coconut] = orders_co
                    result[pina] = orders_pi
                # result[pina] = orders_pi

                elif abs(zscore) < 0.5:

                    if net_position_co > 500 and len(self.prices[coconut]) > 10 and today_price_co > np.mean(
                            self.prices[coconut][-10:]) and best_bid_co:
                        # sell co
                        sell_volume_co = min(abs(net_position_co),
                                             abs(order_depth_co.buy_orders[best_bid_co]))

                        orders_co = [Order(coconut, best_bid_co, -sell_volume_co)]
                        self.positions[coconut] -= sell_volume_co
                        result[coconut] = orders_co

                    if net_position_co < -500 and len(self.prices[coconut]) > 10 and today_price_co < np.mean(
                            self.prices[coconut][-10:]):
                        # buy co
                        buy_volume_co = min(abs(net_position_co),
                                            abs(order_depth_co.sell_orders[best_ask_co]))
                        orders_co = [Order(coconut, best_ask_co, buy_volume_co)]
                        self.positions[coconut] += buy_volume_co
                        result[coconut] = orders_co

                    if net_position_pi > 270 and len(self.prices[pina]) > 10 and today_price_pina > np.mean(
                            self.prices[pina][-10:]) and best_bid_pina:
                        # sell co
                        sell_volume_pi = min(abs(net_position_pi),
                                             abs(order_depth_pi.buy_orders[best_bid_pina]))

                        orders_pi = [Order(pina, best_bid_pina, -sell_volume_pi)]
                        self.positions[pina] -= sell_volume_pi
                        result[pina] = orders_pi

                    if net_position_pi < -270 and len(self.prices[pina]) > 10 and today_price_pina < np.mean(
                            self.prices[pina][-10:]):
                        # buy co
                        buy_volume_pi = min(abs(net_position_pi),
                                            abs(order_depth_pi.sell_orders[best_ask_pina]))
                        orders_pi = [Order(pina, best_ask_pina, buy_volume_pi)]
                        self.positions[pina] += buy_volume_pi
                        result[pina] = orders_pi





            elif product == "BANANAS":

                # Retrieve the Order Depth containing all the market BUY and SELL orders for BANANAS
                order_depth_ba: OrderDepth = state.order_depths[product]

                # Get the current best bid and ask prices
                best_bid = get_best_bid(order_depth_ba)
                best_ask = get_best_ask(order_depth_ba)

                # Calculate the mid price(market price) as the average of the best bid and ask prices
                today_price_ba = get_market_price(best_bid, best_ask)
                if today_price_ba is not None: self.prices["BANANAS"].append(today_price_ba)

                # need historical data
                if len(self.prices["BANANAS"]) < 10:
                    continue

                mid_price = np.mean(self.prices["BANANAS"])
                price_std = np.std(self.prices["BANANAS"])

                # sell short or buy long?????????????
                flag = get_z_score(today_price_ba, mid_price, price_std)

                # Define the acceptable price range as a percentage of the mid price
                acceptable_range = 0.0005  # 5%
                # Define the acceptable buy and sell prices based on the acceptable range
                acceptable_buy_price = math.ceil(today_price_ba * (1 + acceptable_range))
                acceptable_sell_price = math.floor(today_price_ba * (1 - acceptable_range))

                # Define the position limit for BANANAS
                position_limit = self.position_limit[product]
                # Calculate the current net position for BANANAS
                net_position = self.positions[product]

                if verbose_banana_trading:
                    print("-----------BANANA-----------")
                    print("today_price_ba: " + str(today_price_ba))
                    print("mid_price: " + str(mid_price))
                    print("price_std: " + str(price_std))
                    print('banana_zscore: ' + str(flag))
                    print("best_bid: " + str(best_bid))
                    print("best_ask: " + str(best_ask))
                    print("acceptable_buy_price: " + str(acceptable_buy_price))
                    print("acceptable_sell_price: " + str(acceptable_sell_price))
                    print("net_position: " + str(net_position))
                    print("---------------------------")

                # sell short
                if flag >= 1 and best_bid > acceptable_sell_price:
                    max_pos = position_limit + net_position

                    print(max_pos, flag)
                    sell_volume = min(max_pos, abs(order_depth_ba.buy_orders[best_bid]))
                    print("SELL Ba", str(sell_volume) + "x", best_bid)

                    orders = [Order(product, best_bid, -sell_volume)]
                    self.positions[product] -= sell_volume
                    result[product] = orders
                # buy long
                elif flag <= -1 and best_ask < acceptable_buy_price:
                    max_pos = position_limit - net_position

                    buy_volume = min(max_pos, abs(order_depth_ba.sell_orders[best_ask]))
                    print("BUY Ba", str(buy_volume) + "x", best_ask)

                    orders = [Order(product, best_ask, buy_volume)]
                    self.positions[product] += buy_volume
                    result[product] = orders
                # clear position
                elif abs(flag) < 0.5:
                    # sell

                    if net_position > 0 and best_bid and best_bid >= acceptable_buy_price:
                        sell_volume = min(net_position, abs(order_depth_ba.buy_orders[best_bid]))
                        print("SELL Ba", str(sell_volume) + "x", best_bid)
                        orders = [Order(product, best_bid, -sell_volume)]
                        self.positions[product] -= sell_volume
                        result[product] = orders
                    elif net_position < 0 and best_ask and best_ask <= acceptable_sell_price:
                        buy_volume = min(abs(net_position), abs(order_depth_ba.sell_orders[best_ask]))
                        print("BUY Ba", str(buy_volume) + "x", best_ask)
                        orders = [Order(product, best_ask, buy_volume)]
                        self.positions[product] += buy_volume
                        result[product] = orders

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

            # if state.timestamp == 99900:
            #     print(self.temp)

        if visualizer:
            logger.flush(state, result)
        return result
