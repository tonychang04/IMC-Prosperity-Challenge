from typing import Dict, List

import numpy as np

from datamodel import TradingState, Order, Trade


class Trader:
    def __init__(self):
        self.product_params = {}

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {}

        def calculate_fair_value(self, product: str, trades: List[Trade]) -> float:
            # Check if the product's parameters have already been initialized, and if not, initialize them
            if product not in self.product_params:
                self.product_params[product] = {'window_size': 10, 'alpha': 0.2,
                                                'buy_price_improvement': 0.005,
                                                'sell_price_improvement': 0.005,
                                                'max_spread': 0.05}
            window_size = self.product_params[product]['window_size']
            if len(trades) < window_size:
                window_size = len(trades)
            alpha = self.product_params[product]['alpha']
            ema = trades[-window_size].price
            total_quantity = trades[-window_size].quantity
            prices = []
            for trade in trades[-window_size:]:
                quantity_weight = min(trade.quantity / total_quantity, 1.0)
                ema = alpha * trade.price * quantity_weight + (1 - alpha) * ema
                total_quantity += trade.quantity
                prices.append(trade.price)
            fair_value = ema
            return fair_value

        def calculate_momentum_direction(self, product: str, trades: List[Trade]) -> str:
            momentum_direction = "none"
            if len(trades) > 1:
                prices = [trade.price for trade in trades]
                momentum = np.diff(prices)[-1]
                if momentum > 0:
                    momentum_direction = "up"
                elif momentum < 0:
                    momentum_direction = "down"
                else:
                    momentum_direction = "none"
            return momentum_direction

        for product in state.order_depths.keys():
            if product == 'BANANAS':
                # Retrieve the historical price data for the product
                trades: List[Trade] = state.market_trades.get(product, [])
                if len(trades) == 0:
                    continue

                # Calculate the fair value for the product
                fair_value = calculate_fair_value(product, trades)

                # Determine the momentum direction of the asset
                momentum_direction = calculate_momentum_direction(product, trades)

                order_depth = state.order_depths[product]
                if order_depth.buy_orders and order_depth.sell_orders:
                    # Retrieve the best bid and best ask from the order book
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2

                    # Determine the trading strategy
                    order_quantity = min(order_depth.sell_orders[best_ask], order_depth.buy_orders[best_bid])
                    if momentum_direction == "up":
                        # If the asset is exhibiting positive momentum, place a buy order if the mid-price is below
                        # the fair value and the bid-ask spread is not too wide
                        if mid_price < fair_value and (best_ask - best_bid) / mid_price < self.product_params[product]['max_spread']:
                            order_price = mid_price * (1 - self.product_params[product]['buy_price_improvement'])
                            orders = [Order(product, order_price, order_quantity)]
                            result[product] = orders
                            print("BUY", product, str(order_quantity) + "x", order_price)
                        elif momentum_direction == "down":
                            # If the asset is exhibiting negative momentum, place a sell order if the mid-price is above
                            # the fair value and the bid-ask spread is not too wide
                            if mid_price > fair_value and (best_ask - best_bid) / mid_price < \
                                    self.product_params[product]['max_spread']:
                                order_price = mid_price * (1 + self.product_params[product]['sell_price_improvement'])
                                orders = [Order(product, order_price, order_quantity)]
                                result[product] = orders
                                print("SELL", product, str(order_quantity) + "x", order_price)
                            else:
                                # If there is not enough price data to determine momentum or the bid-ask spread is too wide, do nothing
                                pass

        return result


