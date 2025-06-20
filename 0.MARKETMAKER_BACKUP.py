
class MarketMaker:
    def __init__(self, params):
        # Parameters with defaults
        self.tick_size = params.get("tick_size", 1)                   # minimum price increment
        self.inventory_limit = params.get("inventory_limit", 200)     # maximum absolute position allowed
        self.reequote_interval = params.get("requote_interval", 0.02) # seconds between re-quotes
        self.order_quantity = params.get("order_quantity", 5)         # order size
        self.levels = params.get("levels", 5)                         # number of price levels per side
        
        # Ignore balance and only track inventory
        self.position = 0  
        self.last_quote_time = None
        # Instead of single orders, we use lists for active orders
        self.active_buy_orders = []
        self.active_sell_orders = []
        # Store midprices to optionally use an older value if desired
        self.midprice_history = []

    def update_lob(self, lob_data):
        """
        Expects lob_data as a dictionary with keys:
            "BestBid" and "BestAsk"
        """
        best_bid = public_data['bids']['best']
        best_ask = public_data['asks']['best']

        #best_bid = lob_data.get("BestBid", TBSE_SYS_MIN_PRICE)
        #best_ask = lob_data.get("BestAsk", TBSE_SYS_MAX_PRICE)
        return best_bid, best_ask

    def get_midprice(self, best_bid, best_ask):
        """
        Calculate the midprice from the current best bid and ask.
        If both are available, return their average; otherwise return the available price.
        """
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        else:
            return (TBSE_SYS_MIN_PRICE + TBSE_SYS_MAX_PRICE) / 2.0

    def cancel_order(self, order):
        """
        Simulate order cancellation by marking the order inactive.
        """
        order["active"] = False
        print(f"Cancelled order: {order}")

    def place_order(self, side, price, quantity):
        """
        Simulate placing an order. Returns an order dictionary.
        """
        order = {"side": side, "price": price, "quantity": quantity, "active": True}
        print(f"Placed order: {order}")
        return order

    def re_quote(self, lob_data, current_time):
        best_bid, best_ask = self.update_lob(lob_data)
        midprice = self.get_midprice(best_bid, best_ask)
        self.midprice_history.append(midprice)
        # Optionally, use an older midprice if you prefer—for example, 5 events ago:
        if len(self.midprice_history) >= 3:
            effective_mid = self.midprice_history[-3]
        else:
            effective_mid = midprice

        # Build a ladder of desired buy prices.
        buy_prices = []
        for level in range(self.levels):
            # Start with best_bid + tick_size (most aggressive) and then decrement.
            price = best_bid + self.tick_size - level * self.tick_size
            # To ensure we never buy above fair value, clip to midprice.
            price = min(price, midprice - self.tick_size)  
            buy_prices.append(int(price))

        # Build a ladder of desired sell prices.
        sell_prices = []
        for level in range(self.levels):
            # Start with best_ask - tick_size and then increment.
            price = best_ask - self.tick_size + level * self.tick_size
            # To ensure we never sell below fair value, clip to midprice.
            price = max(price, midprice + self.tick_size)
            sell_prices.append(int(price))

        # Safety: if buy ladder overlaps sell ladder, re-adjust around effective_mid.
        if max(buy_prices) >= min(sell_prices):
            buy_prices = [int(effective_mid - self.tick_size - i * self.tick_size) for i in range(self.levels)]
            sell_prices = [int(effective_mid + self.tick_size + i * self.tick_size) for i in range(self.levels)]

        # Cancel any active orders not matching the desired ladder.
        for order in self.active_buy_orders[:]:
            if order["price"] not in buy_prices:
                self.cancel_order(order)
                self.active_buy_orders.remove(order)
        for order in self.active_sell_orders[:]:
            if order["price"] not in sell_prices:
                self.cancel_order(order)
                self.active_sell_orders.remove(order)

        # Place new buy orders for missing price levels (if within inventory limits).
        current_buy_prices = [order["price"] for order in self.active_buy_orders]
        for price in buy_prices:
            if price not in current_buy_prices and self.position > -self.inventory_limit:
                new_order = self.place_order("BUY", price, self.order_quantity)
                self.active_buy_orders.append(new_order)

        # Place new sell orders for missing price levels.
        current_sell_prices = [order["price"] for order in self.active_sell_orders]
        for price in sell_prices:
            if price not in current_sell_prices and self.position < self.inventory_limit:
                new_order = self.place_order("SELL", price, self.order_quantity)
                self.active_sell_orders.append(new_order)

        self.last_quote_time = current_time

    def on_fill(self, order, fill_price):
        """
        When an order is filled, update the maker’s inventory.
        """
        if order["side"] == "BUY":
            self.position += order["quantity"]
        elif order["side"] == "SELL":
            self.position -= order["quantity"]
        order["active"] = False
        print(f"Order filled: {order} at {fill_price}. New position: {self.position}")

    def run(self, lob_stream, market_order_hits):
        """
        Runs the market maker over a stream of LOB events.
        :param lob_stream: Iterable of events (dicts) with keys "Time" and "LOB".
        :param market_order_hits: Function taking (order, lob_data) and returning True if order is filled.
        """
        for event in lob_stream:
            current_time = event["Time"]
            lob_data = event["LOB"]

            # Re-quote if enough time has passed.
            if (self.last_quote_time is None) or (current_time - self.last_quote_time >= self.reequote_interval):
                self.re_quote(lob_data, current_time)

            # Check for fills on active orders.
            for order in self.active_buy_orders[:]:
                if market_order_hits(order, lob_data):
                    self.on_fill(order, lob_data.get("BestAsk"))
                    self.active_buy_orders.remove(order)
            for order in self.active_sell_orders[:]:
                if market_order_hits(order, lob_data):
                    self.on_fill(order, lob_data.get("BestBid"))
                    self.active_sell_orders.remove(order)


