import numpy as np
from typing import Dict, List

from datamodel import TradingState, Order, Trade


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {}

        # Define the fair value for each product
        for product in state.order_depths.keys():
            # Skip any products that don't have both buy and sell orders in the order book
            if not state.order_depths[product].buy_orders or not state.order_depths[product].sell_orders:
                continue

            # Retrieve the historical trade data for the product
            trades: List[Trade] = state.market_trades.get(product, [])
            if len(trades) == 0:
                continue

            # Calculate the fair value for the product using the geometric mean of the last 10 trades
            prices = np.array([trade.price for trade in trades[-10:]])
            fair_value = np.exp(np.mean(np.log(prices)))

            # Calculate the mid-price of the order book
            order_depth = state.order_depths[product]
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            # Determine the total buy and sell volume in the order book
            buy_volume = sum(order_depth.buy_orders.values())
            sell_volume = sum(order_depth.sell_orders.values())

            # Determine the imbalance of the order book as the difference between the total buy and sell volume
            imbalance = buy_volume - sell_volume

            # If the mid-price is below the fair value and the order book is more heavily sell-imbalanced, send a buy order
            if mid_price < fair_value and imbalance < 0:
                # Calculate the quantity to buy as a function of the sell imbalance and the fair value
                quantity = min(abs(imbalance) / (fair_value * 2), order_depth.sell_orders[best_ask])
                price = best_ask
                orders = [Order(product, price, quantity)]
                result[product] = orders
                print("BUY", product, f"{quantity:.2f}x", f"{price:.2f}")

            # If the mid-price is above the fair value and the order book is more heavily buy-imbalanced, send a sell order
            elif mid_price > fair_value and imbalance > 0:
                # Calculate the quantity to sell as a function of the buy imbalance and the fair value
                quantity = min(imbalance / (fair_value * 2), order_depth.buy_orders[best_bid])
                price = best_bid
                orders = [Order(product, price, -quantity)]
                result[product] = orders
                print("SELL", product, f"{quantity:.2f}x", f"{price:.2f}")

        return result
