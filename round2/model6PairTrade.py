from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import numpy as np

class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Define the names of the two products we want to trade
        product1 = "PINA_COLADAS"
        product2 = "COCONUTS"

        # Retrieve the Order Depth containing all the market BUY and SELL orders for the two products
        order_depth1: OrderDepth = state.order_depths[product1]
        order_depth2: OrderDepth = state.order_depths[product2]

        # Initialize the list of Orders to be sent as an empty list
        orders1: list[Order] = []
        orders2: list[Order] = []

        if product1 not in state.market_trades or product2 not in state.market_trades:
            return result

        # Define a fair spread value for the two products， which is the historical average spread
        #fair_spread = 0.0001
        #time = state.timestamp

        state.market_trades[product1].sort(key=lambda x: x.timestamp, reverse=True)
        state.market_trades[product2].sort(key=lambda x: x.timestamp, reverse=True)
        count = 200
        curr = 0
        diff_price = []
        price1 = []
        price2 = []
        for i in range(len(state.market_trades[product1])):


            if i >= len(state.market_trades[product2]):
                break
            diff_price.append(state.market_trades[product1][i].price - state.market_trades[product2][i].price)
            price1.append(state.market_trades[product1][i].price)
            price2.append(state.market_trades[product2][i].price)

            curr += 1
            if curr == count:
                break
        fair_spread = np.mean(diff_price)
        mean_price1 = np.mean(price1)
        std_price1 = np.std(price1)
        mean_price2 = np.mean(price2)
        std_price2 = np.std(price2)

        print("fair_spread", fair_spread)
        print("mean_price1", mean_price1)
        print("mean_price2", mean_price2)

        # If statement checks if there are any SELL orders in the PINA_COLADAS market
        if len(order_depth1.sell_orders) > 0:

            # Sort all the available sell orders by their price,
            # and select only the sell order with the lowest price
            best_ask1 = min(order_depth1.sell_orders.keys())
            best_ask1_volume = order_depth1.sell_orders[best_ask1]

            # Sort all the available buy orders for the COCONUTS by their price,
            # and select only the buy order with the highest price
            best_bid2 = max(order_depth2.buy_orders.keys())
            best_bid2_volume = order_depth2.buy_orders[best_bid2]

            # Calculate the spread between the two products
            spread = best_ask1 - best_bid2
            print("spread1", spread)
            print("best_ask1", best_ask1)
            print("best_bid2", best_bid2)
            # Check if the spread is lower than the fair spread value
            #if spread < -fair_spread and best_ask1 < mean_price1 + std_price1 and best_bid2 > mean_price2 - std_price2:
            if spread < fair_spread - np.std(diff_price):
                # In case the spread is lower than our fair value,
                # this presents an opportunity for us to buy PINA_COLADAS and sell COCONUTS
                # The code below therefore sends a BUY order for PINA_COLADAS at the price level of the ask,
                # with the same quantity, and a SELL order for COCONUTS at the price level of the bid,
                # with the same quantity
                # We expect these orders to trade with the sell order in PINA_COLADAS market
                # and the buy order in COCONUTS market
                print("BUY", product1, str(-best_ask1_volume) + "x", best_ask1)
                orders1.append(Order(product1, best_ask1, -best_ask1_volume))
                print("SELL", product2, str(best_bid2_volume) + "x", best_bid2)
                orders2.append(Order(product2, best_bid2, best_bid2_volume))

        # If statement checks if there are any SELL orders in the COCONUTS market
        if len(order_depth2.sell_orders) > 0:

            # Sort all the available sell orders by their price,
            # and select only the sell order with the lowest price
            best_ask2 = min(order_depth2.sell_orders.keys())
            best_ask2_volume = order_depth2.sell_orders[best_ask2]

            # Sort all the available buy orders for the PINA_COLADAS by their price,
            # and select only the buy order with the highest price
            best_bid1 = max(order_depth1.buy_orders.keys())
            best_bid1_volume = order_depth1.buy_orders[best_bid1]

            # Calculate the spread between the two products
            spread = best_bid1 - best_ask2
            print("spread2", spread)
            print("best_bid1", best_bid1)
            print("best_as k2", best_ask2)

            # Check if the spread is lower than the fair spread value
           # if spread < fair_spread and best_ask2 < mean_price2 + std_price2 and best_bid1 > mean_price1 - std_price1:
            if spread > fair_spread + np.std(diff_price):
                # In case the spread is lower than our fair value,
                # this presents an opportunity for us to buy COCONUTS and sell PINA_COLADAS
                # The code below therefore sends a BUY order for COCONUTS at the price level of the ask,
                # with the same quantity, and a SELL order for PINA_COLADAS at the price level of the bid,
                # with the same quantity
                # We expect these orders to trade with the sell order in COCONUTS market
                # and the buy order in PINA_COLADAS market
                print("BUY", product2, str(-best_ask2_volume) + "x", best_ask2)
                orders2.append(Order(product2, best_ask2, -best_ask2_volume))
                print("SELL", product1, str(best_bid1_volume) + "x", best_bid1)
                orders1.append(Order(product1, best_bid1, best_bid1_volume))

        # Add the list of orders to be sent to the method output dictionary,
        # with the key being the name of the trader class
        result[product1] = orders1
        result[product2] = orders2

        # Return the method output dictionary
        return result
