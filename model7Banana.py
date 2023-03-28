from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math
import numpy as np


class Trader:

    def __init__(self):
        self.positions = {'BANANAS': 0}  # initialize position dictionary
        self.mean_price = []

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        print("net_position of BANANAS: " + str(self.positions['BANANAS']))

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            # Check if the current product is BANANAS, only then run the order logic
            if product == 'BANANAS':

                # Retrieve the Order Depth containing all the market BUY and SELL orders for BANANAS
                order_depth: OrderDepth = state.order_depths[product]

                # Get the current best bid and ask prices
                best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
                best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None

                # Calculate the mid price(market price) as the average of the best bid and ask prices
                today_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
                if today_price is not None: self.mean_price.append(today_price)
                # need historical data
                if len(self.mean_price) < 10: return result
                mid_price = np.mean(self.mean_price)
                print('today price: ' + str(today_price))
                print('mean_price: ' + str(mid_price))
                price_std = np.std(self.mean_price)
                print('price_std: ' + str(price_std))

                # sell short or buy long?????????????
                flag = (today_price - mid_price) / price_std
                print('flag: ' + str(flag))

                # Define the acceptable price range as a percentage of the mid price
                acceptable_range = 0.0005  # 5%
                # Define the acceptable buy and sell prices based on the acceptable range
                acceptable_buy_price = math.ceil(today_price * (1 + acceptable_range))
                acceptable_sell_price = math.floor(today_price * (1 - acceptable_range))
                # Define the position limit for BANANAS
                position_limit = 20
                # Calculate the current net position for BANANAS
                net_position = self.positions[product]

                # sell short
                if flag >= 1 and best_bid > acceptable_sell_price:
                    max_pos = position_limit + net_position if net_position >= 0 else abs(
                        -position_limit - net_position)
                    print('acceptable_sell_price : ' + str(acceptable_sell_price))
                    print('best_bid : ' + str(best_bid))
                    print('amx_pos and flag')
                    print(max_pos, flag)
                    sell_volume = min(max_pos, abs(order_depth.buy_orders[best_bid]))
                    print("SELL", str(sell_volume) + "x", best_bid)

                    orders = [Order(product, best_bid, sell_volume)]
                    self.positions[product] -= sell_volume
                    result[product] = orders
                # buy long
                elif flag <= -1 and best_ask < acceptable_buy_price:
                    max_pos = position_limit - net_position if net_position >= 0 else position_limit + abs(net_position)
                    print('acceptable_buy_price : ' + str(acceptable_buy_price))
                    print('best_ask : ' + str(best_ask))
                    print('amx_pos and flag')
                    print(max_pos, flag)
                    buy_volume = min(max_pos, abs(order_depth.sell_orders[best_ask]))
                    print("BUY", str(buy_volume) + "x", best_ask)
                    orders = [Order(product, best_ask, buy_volume)]
                    self.positions[product] += buy_volume
                    result[product] = orders
                # clear position
                elif abs(flag) < 0.5:
                    # sell
                    print('netposition and flag')
                    print(net_position, flag)
                    if net_position > 0 and best_bid and best_bid >= acceptable_buy_price:
                        sell_volume = min(net_position, abs(order_depth.buy_orders[best_bid]))
                        print("SELL", str(sell_volume) + "x", best_bid)
                        orders = [Order(product, best_bid, sell_volume)]
                        self.positions[product] -= sell_volume
                        result[product] = orders
                    elif net_position < 0 and best_ask and best_ask <= acceptable_sell_price:
                        buy_volume = min(abs(net_position), abs(order_depth.sell_orders[best_ask]))
                        print("BUY", str(buy_volume) + "x", best_ask)
                        orders = [Order(product, best_ask, buy_volume)]
                        self.positions[product] += buy_volume
                        result[product] = orders

        return result
