from typing import Dict, List

from datamodel import TradingState, Order, Trade
import numpy as np

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {}

        # Define the fair value for product
        for product in state.order_depths.keys():
            if product == 'BANANAS':
            #  orders: list[Order] = []
            # Check if there are any buy or sell orders for the PEARLS product

            # Retrieve the historical price data for the product
                trades: List[Trade] = state.market_trades.get(product, [])
                if len(trades) == 0:
                    continue
                trades.sort(key=lambda x: x.timestamp)

                # Calculate the fair value for the product, which is the moving average
                # of the last 20 trades
                window_size = 20
                if len(trades) < window_size:
                    window_size = len(trades)
                alpha = 2 / (window_size + 1)
                ema = trades[-window_size].price
                total_quantity = trades[-window_size].quantity
                prices = []
                for trade in trades[-window_size:]:
                    quantity_weight = min(trade.quantity / total_quantity, 1.0)
                    ema = alpha * trade.price * quantity_weight + (1 - alpha) * ema
                    total_quantity += trade.quantity
                    prices.append(trade.price)
                fair_value = ema

                order_depth = state.order_depths[product]
                if order_depth.buy_orders and order_depth.sell_orders:
                    # Retrieve the best bid and best ask from the order book
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())

                    # Calculate the mid-price
                   # mid_price = (best_bid + best_ask) / 2

                    # Calculate the total buy and sell volume
                    buy_volume = sum(order_depth.buy_orders.values())
                    sell_volume = sum(order_depth.sell_orders.values())

                    # Calculate the percentage of the total trading volume that is a buy or sell order
                    buy_percentage = (buy_volume / (buy_volume + sell_volume)) if buy_volume + sell_volume > 0 else 0
                    sell_percentage = (sell_volume / (buy_volume + sell_volume)) if buy_volume + sell_volume > 0 else 0

                    # Determine the threshold for the buy and sell orders based on the percentage of the total trading volume
                    buy_threshold = 0.5 + 0.2 * sell_percentage
                    sell_threshold = 0.5 + 0.2 * buy_percentage
                    # If the mid-price is less than the fair value, send a buy order
                    if best_ask < fair_value - 5*np.sqrt(np.var(prices)) and buy_percentage < buy_threshold:
                        orders = [Order(product, best_ask, order_depth.sell_orders[best_ask])]
                        result[product] = orders


                        print("BUY", product, str(order_depth.sell_orders[best_ask]) + "x", best_ask)

                    # If the mid-price is greater than the fair value, send a sell order
                    elif best_bid > fair_value + 5*np.sqrt(np.var(prices)) and sell_percentage < sell_threshold:
                        orders = [Order(product, best_bid, -order_depth.buy_orders[best_bid])]
                        result[product] = orders
                        print("SELL", product, str(order_depth.buy_orders[best_bid]) + "x", best_bid)

        # result[product] = orders

        return result
