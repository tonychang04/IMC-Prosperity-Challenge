from typing import Dict, List

import numpy as np

from datamodel import TradingState, Order, Trade


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {}
        return result
        '''
        for product in state.order_depths.keys():
            if product == 'BANANAS':
                # Retrieve the historical price data for the product
                trades: List[Trade] = state.market_trades.get(product, [])
                if len(trades) == 0:
                    continue
                trades.sort(key=lambda x: x.timestamp)

                # Calculate the fair value for the product, which is the moving average
                # of the last 10 trades
                window_size = 10
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
                    mid_price = (best_bid + best_ask) / 2

                    # Determine the momentum direction of the asset
                    if len(prices) > 1:
                        momentum = np.diff(prices)[-1]
                        if momentum > 0:
                            # If the asset is exhibiting positive momentum, place a buy order
                            if mid_price < fair_value:
                                # Only place a buy order if the price has decreased by more than 1%
                                price_change = (fair_value - mid_price) / mid_price
                                if price_change > 0.01:
                                    # Buy at the best ask price with all available funds

                                    quantity = order_depth.sell_orders[best_ask]
                                    orders = [Order(product, best_ask, quantity)]
                                    result[product] = orders
                                    print("BUY", product, str(quantity) + "x", best_ask)
                        elif momentum < 0:
                            # If the asset is exhibiting negative momentum, place a sell order
                            if mid_price > fair_value:
                                # Only place a sell order if the price has increased by more than 1%
                                price_change = (mid_price - fair_value) / fair_value
                                if price_change > 0.01:
                                    # Sell all held units at the best bid price
                                    quantity = order_depth.buy_orders[best_bid]
                                    orders = [Order(product, best_bid, quantity)]
                                    result[product] = orders
                                    print("SELL", product, str(quantity) + "x", best_bid)
                        else:
                            # If there is not enough price data to determine momentum, do nothing
                            pass

        return result
        '''
