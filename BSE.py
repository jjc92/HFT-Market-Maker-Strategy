import sys
import math
import random
import os
import time as chrono
import numpy as np
money = 0

# a bunch of system constants (globals)
bse_sys_minprice = 1                    # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500                  # maximum price in the system, in cents/pennies
# ticksize should be a param of an exchange (so different exchanges have different ticksizes)
ticksize = 1  # minimum change in price, in cents/pennies

# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:

    def __init__(self, tid, otype, price, qty, time, qid):
        self.tid = tid  # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty  # quantity
        self.time = time  # timestamp
        self.qid = qid  # quote i.d. (unique to each quote)

    def __str__(self):
        return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
               (self.tid, self.otype, self.price, self.qty, self.time, self.qid)


# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

    def __init__(self, booktype, worstprice):
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worstprice = worstprice
        self.session_extreme = None    # most extreme price quoted in this session
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        # anonymize a lob, strip out order details, format as a sorted list
        # NB for asks, the sorting should be reversed
        self.lob_anon = []
        for price in sorted(self.lob):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        lob_verbose = False
        # take a list of orders and build a limit-order-book (lob) from it
        # NB the exchange needs to know arrival times and trader-id associated with each order
        # returns lob as a dictionary (i.e., unsorted)
        # also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        self.lob = {}
        for tid in self.orders:
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                orderlist = self.lob[price][1]
                orderlist.append([order.time, order.qty, order.tid, order.qid])
                self.lob[price] = [qty + order.qty, orderlist]
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.booktype == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        # add order to the dictionary holding the list of orders
        # either overwrites old order from this trader
        # or dynamically creates new entry in the dictionary
        # so, max of one order per trader per list
        # checks whether length or order list has changed, to distinguish addition/overwrite
        # print('book_add > %s %s' % (order, self.orders))

        # if this is an ask, does the price set a new extreme-high record?
        if (self.booktype == 'Ask') and ((self.session_extreme is None) or (order.price > self.session_extreme)):
            self.session_extreme = int(order.price)

        # add the order to the book
        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()
        # print('book_add < %s %s' % (order, self.orders))
        if n_orders != self.n_orders:
            return 'Addition'
        else:
            return 'Overwrite'

    def book_del(self, order):
        # delete order from the dictionary holding the orders
        # assumes max of one order per trader per list
        # checks that the Trader ID does actually exist in the dict before deletion
        # print('book_del %s',self.orders)
        if self.orders.get(order.tid) is not None:
            del (self.orders[order.tid])
            self.n_orders = len(self.orders)
            self.build_lob()
        # print('book_del %s', self.orders)

    def delete_best(self):
        # delete order: when the best bid/ask has been hit, delete it from the book
        # the TraderID of the deleted order is return-value, as counterparty to the trade
        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del (self.lob[self.best_price])
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.booktype == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worstprice
                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty


# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

    def __init__(self):
        self.bids = Orderbook_half('Bid', bse_sys_minprice)
        self.asks = Orderbook_half('Ask', bse_sys_maxprice)
        self.tape = []
        self.tape_length = 1000000000  # max number of events on tape (so we can do millions of orders without crashing)
        self.quote_id = 0           # unique ID code for each quote accepted onto the book
        self.lob_string = ''        # character-string linearization of public lob items with nonzero quantities


# Exchange's internal orderbook

class Exchange(Orderbook):

    def add_order(self, order, verbose):
        # add a quote/order to the exchange and update all internal records; return unique i.d.
        order.qid = self.quote_id
        self.quote_id = order.qid + 1
        # if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.qid, response]

    def del_order(self, time, order, verbose):
        # delete a trader's quot/order from the exchange, update all internal records
        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    def process_order2(self, time, order, verbose):
        # receive an order and either add it to the relevant LOB (ie treat as limit order)
        # or if it crosses the best counterparty offer, execute it (treat as a market order)
        oprice = order.price
        counterparty = None
        price = None
        [qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
        order.qid = qid
        if verbose:
            print('QUID: order.quid=%d' % order.qid)
            print('RESPONSE: %s' % response)
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid
        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:
                # bid lifts the best ask
                if verbose:
                    print("Bid $%s lifts best ask" % oprice)
                counterparty = best_ask_tid
                price = best_ask  # bid crossed ask, so use ask price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # ask hits the best bid
                if verbose:
                    print("Ask $%s hits best bid" % oprice)
                # remove the best bid
                counterparty = best_bid_tid
                price = best_bid  # ask crossed bid, so use bid price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()
        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if verbose:
            print('counterparty %s' % counterparty)
        if counterparty is not None:
            # process the trade
            if verbose:
                print('>>>>>>>>>>>>>>>>>TRADE t=%010.3f $%d %s %s' % (time, price, counterparty, order.tid))
            transaction_record = {'type': 'Trade',
                                  'time': time,
                                  'price': price,
                                  'party1': counterparty,
                                  'party2': order.tid,
                                  'qty': order.qty
                                  }
            self.tape.append(transaction_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

            return transaction_record
        else:
            return None

    # Currently tape_dump only writes a list of transactions (ignores cancellations)
    def tape_dump(self, fname, fmode, tmode):
        dumpfile = open(fname, fmode)
        # dumpfile.write('type, time, price\n')
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                dumpfile.write('Trd, %010.3f, %s\n' % (tapeitem['time'], tapeitem['price']))
        dumpfile.close()
        if tmode == 'wipe':
            self.tape = []

    # this returns the LOB data "published" by the exchange,
    # i.e., what is accessible to the traders
    def publish_lob(self, time, lob_file, verbose):
        public_data = {}
        public_data['time'] = time
        public_data['bids'] = {'best': self.bids.best_price,
                               'worst': self.bids.worstprice,
                               'n': self.bids.n_orders,
                               'lob': self.bids.lob_anon}
        public_data['asks'] = {'best': self.asks.best_price,
                               'worst': self.asks.worstprice,
                               'sess_hi': self.asks.session_extreme,
                               'n': self.asks.n_orders,
                               'lob': self.asks.lob_anon}
        public_data['QID'] = self.quote_id
        public_data['tape'] = self.tape

        if lob_file is not None:
            # build a linear character-string summary of only those prices on LOB with nonzero quantities
            lobstring = 'Bid:,'
            n_bids = len(self.bids.lob_anon)
            if n_bids > 0:
                lobstring += '%d,' % n_bids
                for lobitem in self.bids.lob_anon:
                    price_str = '%d,' % lobitem[0]
                    qty_str = '%d,' % lobitem[1]
                    lobstring = lobstring + price_str + qty_str
            else:
                lobstring += '0,'
            lobstring += 'Ask:,'
            n_asks = len(self.asks.lob_anon)
            if n_asks > 0:
                lobstring += '%d,' % n_asks
                for lobitem in self.asks.lob_anon:
                    price_str = '%d,' % lobitem[0]
                    qty_str = '%d,' % lobitem[1]
                    lobstring = lobstring + price_str + qty_str
            else:
                lobstring += '0,'
            # is this different to the last lob_string?
            if lobstring != self.lob_string:
                # write it
                lob_file.write('%.3f, %s\n' % (time, lobstring))
                # remember it
                self.lob_string = lobstring

        if verbose:
            print('publish_lob: t=%d' % time)
            print('BID_lob=%s' % public_data['bids']['lob'])
            # print('best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders))
            print('ASK_lob=%s' % public_data['asks']['lob'])
            # print('qid=%d' % self.quote_id)

        return public_data


# #################--Traders below here--#############


# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:

    def __init__(self, ttype, tid, balance, params, time):
        self.ttype = ttype          # what type / strategy this trader is
        self.tid = tid              # trader unique ID code
        self.balance = balance      # money in the bank
        self.params = params        # parameters/extras associated with this trader-type or individual trader.
        self.blotter = []           # record of trades executed
        self.blotter_length = 1000000000   # maximum length of blotter
        self.orders = []            # customer orders currently being worked (fixed at len=1 in BSE1.x)
        self.n_quotes = 0           # number of quotes live on LOB
        self.birthtime = time       # used when calculating age of a trader/strategy
        self.profitpertime = 0      # profit per unit time
        self.profit_mintime = 60    # minimum duration in seconds for calculating profitpertime
        self.n_trades = 0           # how many trades has this trader done?
        self.lastquote = None       # record of what its last quote was

    def __str__(self):
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
               % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)

    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose:
            print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        self.orders = []

    def profitpertime_update(self, time, birthtime, totalprofit):
        time_alive = (time - birthtime)
        if time_alive >= self.profit_mintime:
            profitpertime = totalprofit / time_alive
        else:
            # if it's not been alive long enough, divide it by mintime instead of actual time
            profitpertime = totalprofit / self.profit_mintime
        return profitpertime

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        #if profit < 0:
            #print(profit)
            #print(trade)
            #print(order)
            #sys.exit('FAIL: negative profit')

        if verbose:
            print('%s profit=%d balance=%d profit/time=%s' % (outstr, profit, self.balance, str(self.profitpertime)))
        self.del_order(order)  # delete the order

        # if the trader has multiple strategies (e.g. PRSH/PRDE/ZIPSH/ZIPDE) then there is more work to do...
        if hasattr(self, 'strats') and self.strats is not None:
            self.strats[self.active_strat]['profit'] += profit
            totalprofit = self.strats[self.active_strat]['profit']
            birthtime = self.strats[self.active_strat]['start_t']
            self.strats[self.active_strat]['pps'] = self.profitpertime_update(time, birthtime, totalprofit)

    # specify how trader responds to events in the market
    # this is a null action, expect it to be overloaded by specific algos
    def respond(self, time, lob, trade, verbose):
        # any trader subclass with custom respond() must include this update of profitpertime
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)
        return None

    # specify how trader mutates its parameter values
    # this is a null action, expect it to be overloaded by specific algos
    def mutate(self, time, lob, trade, verbose):
        return None


### CREATING HUMAN TYPE TRADER (CAN BUY AND SELL)

class Trader_Human(Trader):
    def __init__(self, ttype, tid, balance, params, time):
        super().__init__(ttype, tid, balance, params, time)
        self.initial_price = None  # Initial price based on market
        self.patience = random.uniform(0.9, 1.2)  # How long they wait before dropping prices
        self.aggressiveness = random.uniform(0.01, 0.05)  # Rate of price adjustment
        self.last_market_price = None  # Track the last known market price
        self.initial_balance = 0
        self.role = 'Seller'  # Initial role, can be 'Seller' or 'Buyer'

    def calculate_initial_price(self, lob):
        """Set the initial price based on market conditions."""
        if self.role == 'Seller':
            if lob['bids']['n'] > 0:  # There are buyers
                return lob['bids']['best'] + ticksize  # Slightly above the best bid
            else:
                return random.randint(bse_sys_minprice, bse_sys_maxprice // 2)  # Default to midpoint
        else:  # Buyer
            if lob['asks']['n'] > 0:  # There are sellers
                return lob['asks']['best'] - ticksize  # Slightly below the best ask
            else:
                return random.randint(bse_sys_minprice, bse_sys_maxprice // 2)  # Default to midpoint

    def adjust_price(self, time_elapsed):
        """Adjust the price based on patience and elapsed time."""
        adjustment = self.aggressiveness * (1 - self.patience / max(time_elapsed, 1))
        if self.role == 'Seller':
            self.orders[0].price = max(self.orders[0].price - int(adjustment * self.orders[0].price), bse_sys_minprice)
        else:  # Buyer
            self.orders[0].price = min(self.orders[0].price + int(adjustment * self.orders[0].price), bse_sys_maxprice)

    def getorder(self, time, countdown, lob):
        """Generate an order based on market conditions."""
        if len(self.orders) < 1:  # No active orders
            return None

        if self.initial_price is None:  # Set the initial price
            self.initial_price = self.calculate_initial_price(lob)
            self.orders[0].price = self.initial_price

        # Adjust the price dynamically over time
        time_elapsed = time - self.birthtime
        self.adjust_price(time_elapsed)

        # Ensure the price does not go below intrinsic value for sellers or above for buyers
        if self.role == 'Seller':
            quote_price = max(self.orders[0].price, bse_sys_minprice)
        else:  # Buyer
            quote_price = min(self.orders[0].price, bse_sys_maxprice)

        # Create and return an order object
        order = Order(self.tid,
                      self.orders[0].otype,
                      quote_price,
                      self.orders[0].qty,
                      time, lob['QID'])
        self.lastquote = order
        return order

    def switch_role(self, new_role):
        """Switch the role of the trader between 'Seller' and 'Buyer'."""
        if new_role in ['Seller', 'Buyer']:
            self.role = new_role
        else:
            raise ValueError("Invalid role. Role must be 'Seller' or 'Buyer'.")

    def record_trade(self, trade):
        """Record a trade to keep track of market trends."""
        self.past_trades.append(trade)
        if len(self.past_trades) > 100:  # Keep the list of past trades manageable
            self.past_trades.pop(0)

#####
#Standardised seller
####

class Trader_Seller(Trader):
    def __init__(self, ttype, tid, balance, params, time):
        super().__init__(ttype, tid, balance, params, time)
        self.initial_price = None  # Initial price based on market
        self.patience = random.uniform(4, 5)  # How long they wait before dropping prices
        self.aggressiveness = random.uniform(0.01, 0.05)  # Rate of price adjustment
        self.last_market_price = None  # Track the last known market price

    def calculate_initial_price(self, lob):
        """Set the initial price based on market conditions."""
        if lob['bids']['n'] > 0:  # There are buyers
            return lob['bids']['best'] + ticksize  # Slightly above the best bid
        else:
            return random.randint(bse_sys_minprice, bse_sys_maxprice // 2)  # Default to midpoint

    def adjust_price(self, time_elapsed):
        """Adjust the price based on patience and elapsed time."""
        adjustment = self.aggressiveness * (1 - self.patience / max(time_elapsed, 1))
        self.orders[0].price = max(self.orders[0].price - int(adjustment * self.orders[0].price), bse_sys_minprice)

    def getorder(self, time, countdown, lob):
        """Generate an order based on market conditions."""
        if len(self.orders) < 1:  # No active orders
            order = None
        else:
            if self.initial_price is None:  # Set the initial price
                self.initial_price = self.calculate_initial_price(lob)
                self.orders[0].price = self.initial_price

            # Adjust the price dynamically over time
            time_elapsed = time - self.birthtime
            self.adjust_price(time_elapsed)

            # Ensure the price does not go below intrinsic value
            quote_price = max(self.orders[0].price, bse_sys_minprice)
            order = Order(
                self.tid,
                self.orders[0].otype,
                quote_price,
                self.orders[0].qty,
                time,
                lob['QID']
            )
            self.lastquote = order
        return order

    def respond(self, time, lob, trade, verbose=False):
        if not self.orders:
            #print(f"WARNING: No orders for trader {self.tid} at time {time}")
            return
        """Update the seller's strategy based on market activity."""
        if trade:  # A trade occurred
            self.last_market_price = trade['price']  # Track last transaction price
        else:  # No trade, observe the LOB
            if lob['bids']['n'] > 0:
                self.last_market_price = lob['bids']['best']  # Update based on best bid

        # Adjust patience and aggressiveness dynamically
        if self.last_market_price:
            if self.last_market_price < self.orders[0].price:
                # Lower patience slightly if priced too high
                self.patience *= 0.99
            else:
                # Increase patience if market supports higher prices
                self.patience *= 1.01

            # Keep patience within bounds
            self.patience = min(1.5, max(0.5, self.patience))



# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)

class Trader_Giveaway(Trader):
    initial_balance = money
    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order 

# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(Trader):
    initial_balance = money
    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            minprice = lob['bids']['worst']
            maxprice = lob['asks']['worst']
            qid = lob['QID']
            limit = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                quoteprice = random.randint(int(minprice), int(limit))
            else:
                quoteprice = random.randint(int(limit), int(maxprice))
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order
        return order


# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(Trader):
    initial_balance = money
    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + 1
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - 1
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(Trader):
    initial_balance = money
    def getorder(self, time, countdown, lob):
        lurk_threshold = 0.2
        shavegrowthrate = 3
        shave = int(1.0 / (0.01 + countdown / (shavegrowthrate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + shave
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - shave
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass PRZI (ticker: PRSH)
# added 6 Sep 2022 -- replaces old PRZI and PRZI_SHC, unifying them into one function and also adding PRDE
#
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader -- pronounced "prezzie"
# but with added adaptive strategies, currently either...
#   ++ a k-point Stochastic Hill-Climber (SHC) hence PRZI-SHC,
#      PRZI-SHC pronounced "prezzy-shuck". Ticker symbol PRSH pronounced "purrsh";
# or
#   ++ a simple differential evolution (DE) optimizer with pop_size=k, hence PRZE-DE or PRDE ('purdy")
#
# when optimizer == None then it implements plain-vanilla non-adaptive PRZI, with a fixed strategy-value.

class Trader_PRZI(Trader):
    initial_balance = money
    # return strategy as a csv-format string (trivial in PRZI, but other traders with more complex strategies need this)
    def strat_csv_str(self, strat):
        csv_str = 's=,%+5.3f, ' % strat
        return csv_str

    # how to mutate the strategy values when evolving / hill-climbing
    def mutate_strat(self, s, mode):
        s_min = self.strat_range_min
        s_max = self.strat_range_max
        if mode == 'gauss':
            sdev = 0.05
            newstrat = s
            while newstrat == s:
                newstrat = s + random.gauss(0.0, sdev)
                # truncate to keep within range
                newstrat = max(-1.0, min(1.0, newstrat))
        elif mode == 'uniform_whole_range':
            # draw uniformly from whole range
            newstrat = random.uniform(-1.0, +1.0)
        elif mode == 'uniform_bounded_range':
            # draw uniformly from bounded range
            newstrat = random.uniform(s_min, s_max)
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return newstrat

    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = '%s: %s active_strat=[%d]:\n' % (self.tid, self.ttype, self.active_strat)
        for s in range(0, self.k):
            strat = self.strats[s]
            stratstr = '[%d]: s=%+f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string

    def __init__(self, ttype, tid, balance, params, time):
        # if params == "landscape-mapper" then it generates data for mapping the fitness landscape

        verbose = True

        Trader.__init__(self, ttype, tid, balance, params, time)

        # unpack the params
        # for all three of PRZI, PRSH, and PRDE params can include strat_min and strat_max
        # for PRSH and PRDE params should include values for optimizer and k
        # if no params specified then defaults to PRZI with strat values in [-1.0,+1.0]

        # default parameter values
        k = 1
        optimizer = None    # no optimizer => plain non-adaptive PRZI
        s_min = -1.0
        s_max = +1.0

        # did call provide different params?
        if type(params) is dict:
            if 'k' in params:
                k = params['k']
            if 'optimizer' in params:
                optimizer = params['optimizer']
            s_min = params['strat_min']
            s_max = params['strat_max']

        self.optmzr = optimizer     # this determines whether it's PRZI, PRSH, or PRDE
        self.k = k                  # number of sampling points (cf number of arms on a multi-armed-bandit, or pop-size)
        self.theta0 = 100           # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.strat_wait_time = 7200     # how many secs do we give any one strat before switching?
        self.strat_range_min = s_min    # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = s_max    # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None        # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time   # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.0 * random.random()    # minimum profit-per-sec difference between strategies that counts
        self.strats = []            # strategies awaiting initialization
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1, 10))  # multiplier coefficient when estimating p_max
        self.mapper_outfile = None
        # differential evolution parameters all in one dictionary
        self.diffevol = {'de_state': 'active_s0',          # initial state: strategy 0 is active (being evaluated)
                         's0_index': self.active_strat,    # s0 starts out as active strat
                         'snew_index': self.k,             # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,            # assigned later
                         'F': 0.8                          # differential weight -- usually between 0 and 2
                        }

        start_time = time
        profit = 0.0
        profit_per_second = 0
        lut_bid = None
        lut_ask = None

        for s in range(self.k + 1):
            # initialise each of the strategies in sequence:
            # for PRZI: only one strategy is needed
            # for PRSH, one random initial strategy, then k-1 mutants of that initial strategy
            # for PRDE, use draws from uniform distbn over whole range and a (k+1)th strategy is needed to hold s_new
            strategy = None
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                if self.optmzr == 'PRSH':
                    # simple stochastic hill climber: cluster other strats around strat_0
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'gauss')     # mutant of strats[0]
                elif self.optmzr == 'PRDE':
                    # differential evolution: seed initial strategies across whole space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_bounded_range')
                else:
                    # plain PRZI -- do nothing
                    pass
            # add to the list of strategies
            if s == self.active_strat:
                active_flag = True
            else:
                active_flag = False
            self.strats.append({'stratval': strategy, 'start_t': start_time, 'active': active_flag,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
            if self.optmzr is None:
                # PRZI -- so we stop after one iteration
                break
            elif self.optmzr == 'PRSH' and s == self.k - 1:
                # PRSH -- doesn't need the (k+1)th strategy
                break

        if self.params == 'landscape-mapper':
            # replace seed+mutants set of strats with regularly-spaced strategy values over the whole range
            self.strats = []
            strategy_delta = 0.01
            strategy = -1.0
            k = 0
            self.strats = []

            while strategy <= +1.0:
                self.strats.append({'stratval': strategy, 'start_t': start_time, 'active': False,
                                    'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
                k += 1
                strategy += strategy_delta
            self.mapper_outfile = open('landscape_map.csv', 'w')
            self.k = k
            self.strat_eval_time = self.k * self.strat_wait_time

        if verbose:
            print("%s\n" % self.strat_str())

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + ticksize   # BSE ticksize is global var
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - ticksize   # BSE ticksize is global var
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            # print('shvr_p=%f; ' % shvr_p)
            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # strat is strategy-value in [-1,+1]
            # t0 and m are constants used in the threshold function
            # dirn is direction: 'buy' or 'sell'
            # pmin and pmax are bounds on discrete-valued price-range

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1*theta0, min(theta0, x))
                return t

            epsilon = 0.000001  # used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('PRSH FAIL: strat=%f out of range\n' % strat)

            if (dirn != 'buy') and (dirn != 'sell'):
                # out of range
                sys.exit('PRSH FAIL: bad dirn=%s\n' % dirn)

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax %f < pmin %f \n' % (pmax, pmin))

            if verbose:
                print('PRSH calc_cdf_lut: strat=%f dirn=%d pmin=%d pmax=%d\n' % (strat, dirn, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the limit-price with probability 1

                if dirn == 'buy':
                    cdf = [{'price': pmax, 'cum_prob': 1.0}]
                else:   # must be a sell
                    cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                # normalize the price to proportion of its range
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strat > 0:
                    if dirn == 'buy':
                        cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                    else:   # dirn == 'sell'
                        cal_p = (math.exp(c * (1 - p_r)) - 1.0) / e2cm1
                else:   # self.strat < 0
                    if dirn == 'buy':
                        cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                    else:   # dirn == 'sell'
                        cal_p = 1.0 - ((math.exp(c * (1 - p_r)) - 1.0) / e2cm1)

                if cal_p < 0:
                    cal_p = 0   # just in case

                calp_interval.append({'price': p, "cal_p": cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

        verbose = False

        if verbose:
            print('t=%.1f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                pass

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible as defined by exchange

            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax    # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5)     # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']         # so use that as my new estimate of highest
                    self.pmax = maxprice

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            if otype == 'Bid':

                p_max = int(limit)
                if strat > 0.0:
                    p_min = minprice
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * minprice))

                lut_bid = self.strats[self.active_strat]['lut_bid']
                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or\
                        (lut_bid['pmin'] != p_min) or \
                        (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = calc_cdf_lut(strat, self.theta0, self.m, 'buy', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:   # otype == 'Ask'

                p_min = int(limit)
                if strat > 0.0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * maxprice))
                    if p_max < p_min:
                        # this should never happen, but just in case it does...
                        p_max = p_min

                lut_ask = self.strats[self.active_strat]['lut_ask']
                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.strats[self.active_strat]['lut_ask'] = calc_cdf_lut(strat, self.theta0, self.m, 'sell', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_ask']

            verbose = False
            if verbose:
                print('PRZI strat=%f LUT=%s \n \n' % (strat, lut))
                # useful in debugging: print a table of lut: price and cum_prob, with the discrete derivative (gives PMF).
                last_cprob = 0.0
                for lut_entry in lut['cdf_lut']:
                    cprob = lut_entry['cum_prob']
                    print('%d, %f, %f' % (lut_entry['price'], cprob - last_cprob, cprob))
                    last_cprob = cprob
                print('\n')
                
                # print ('[LUT print suppressed]')
            
            # do inverse lookup on the LUT to find the price
            quoteprice = None
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]      # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if verbose:
            print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        self.strats[self.active_strat]['profit'] += profit
        time_alive = time - self.strats[self.active_strat]['start_t']
        if time_alive > 0:
            profit_per_second = self.strats[self.active_strat]['profit'] / time_alive
            self.strats[self.active_strat]['pps'] = profit_per_second
        else:
            # if it trades at the instant it is born then it would have infinite profit-per-second, which is insane
            # to keep things sensible when time_alive == 0 we say the profit per second is whatever the actual profit is
            self.strats[self.active_strat]['pps'] = profit

    # PRSH respond() asks/answers two questions
    # do we need to choose a new strategy? (i.e. have just completed/cancelled previous customer order)
    # do we need to dump one arm and generate a new one? (i.e., both/all arms have been evaluated enough)
    def respond(self, time, lob, trade, verbose):

        # "PRSH" is a very basic form of stochastic hill-climber (SHC) that's v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other k-1 strats
        # then all counters are reset, and this is repeated indefinitely
        #
        # "PRDE" uses a basic form of Differential Evolution. This maintains a population of at least four strats
        # iterates indefinitely on:
        #       shuffle the set of strats;
        #       name the first four strats s0 to s3;
        #       create new_strat=s1+f*(s2-s3);
        #       evaluate fitness of s0 and new_strat;
        #       if (new_strat fitter than s0) then new_strat replaces s0.
        #
        # todo: add in other optimizer algorithms that are cleverer than these
        #  e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        def strat_activate(t, s_index):
            # print('t=%f Strat_activate, index=%d, active=%s' % (t, s_index, self.strats[s_index]['active'] ))
            self.strats[s_index]['start_t'] = t
            self.strats[s_index]['active'] = True
            self.strats[s_index]['profit'] = 0.0
            self.strats[s_index]['pps'] = 0.0

        verbose = False

        # first update each active strategy's profit-per-second (pps) value -- this is the "fitness" of each strategy
        for s in self.strats:
            # debugging check: make profit be directly proportional to strategy, no noise
            # s['profit'] = 100 * abs(s['stratval'])
            # update pps
            active_flag = s['active']
            if active_flag:
                s['pps'] = self.profitpertime_update(time, s['start_t'], s['profit'])

        if self.optmzr == 'PRSH':

            if verbose:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                pass

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy
                self.strats[s]['active'] = False

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.strats[new_strat]['active'] = True
                self.last_strat_change_time = time

                if verbose:
                    print('t=%.3fsec (%.2fdays), %s PRSH respond: strat[%d] elapsed=%.3f; wait_t=%.3f, pps=%f,  switched to strat=%d' %
                          (time, time/86400, self.tid, s, time_elapsed, self.strat_wait_time, self.strats[s]['pps'], new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            # assume that all strats have had long enough, and search for evidence to the contrary
            all_old_enough = True
            for s in self.strats:
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key=lambda k: k['pps'], reverse=True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if verbose:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

                if self.params == 'landscape-mapper':
                    for s in self.strats:
                        self.mapper_outfile.write('time, %f, strat, %f, pps, %f\n' %
                                                  (time, s['stratval'], s['pps']))
                    self.mapper_outfile.flush()
                    sys.exit()

                else:
                    # if the difference between the top two strats is too close to call then flip a coin
                    # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                    best_strat = 0
                    prof_diff = strats_sorted[0]['pps'] - strats_sorted[1]['pps']
                    if abs(prof_diff) < self.profit_epsilon:
                        # they're too close to call, so just flip a coin
                        best_strat = random.randint(0, 1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = strats_sorted[0]
                        strats_sorted[0] = strats_sorted[1]
                        strats_sorted[1] = tmp_strat

                    # the sorted list of strats replaces the existing list
                    self.strats = strats_sorted

                    # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate

                    # now replicate and mutate the elite into all the other strats
                    for s in range(1, self.k):    # note range index starts at one not zero (elite is at [0])
                        self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'], 'gauss')
                        self.strats[s]['start_t'] = time
                        self.strats[s]['profit'] = 0.0
                        self.strats[s]['pps'] = 0.0
                    # and then update (wipe) records for the elite
                    self.strats[0]['start_t'] = time
                    self.strats[0]['profit'] = 0.0
                    self.strats[0]['pps'] = 0.0
                    self.active_strat = 0

                if verbose:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

        elif self.optmzr == 'PRDE':
            # simple differential evolution

            # only initiate diff-evol once the active strat has been evaluated for long enough
            actv_lifetime = time - self.strats[self.active_strat]['start_t']
            if actv_lifetime >= self.strat_wait_time:

                if self.k < 4:
                    sys.exit('FAIL: k too small for diffevol')

                if self.diffevol['de_state'] == 'active_s0':
                    self.strats[self.active_strat]['active'] = False
                    # we've evaluated s0, so now we need to evaluate s_new
                    self.active_strat = self.diffevol['snew_index']
                    strat_activate(time, self.active_strat)

                    self.diffevol['de_state'] = 'active_snew'

                elif self.diffevol['de_state'] == 'active_snew':
                    # now we've evaluated s_0 and s_new, so we can do DE adaptive step
                    if verbose:
                        print('PRDE trader %s' % self.tid)
                    i_0 = self.diffevol['s0_index']
                    i_new = self.diffevol['snew_index']
                    fit_0 = self.strats[i_0]['pps']
                    fit_new = self.strats[i_new]['pps']

                    if verbose:
                        print('DiffEvol: t=%.1f, i_0=%d, i0fit=%f, i_new=%d, i_new_fit=%f' % (time, i_0, fit_0, i_new, fit_new))

                    if fit_new >= fit_0:
                        # new strat did better than old strat0, so overwrite new into strat0
                        self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    # do differential evolution

                    # pick four individual strategies at random, but they must be distinct
                    stratlist = list(range(0, self.k))    # create sequential list of strategy-numbers
                    random.shuffle(stratlist)             # shuffle the list

                    # s0 is next iteration's candidate for possible replacement
                    self.diffevol['s0_index'] = stratlist[0]

                    # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
                    s1_index = stratlist[1]
                    s2_index = stratlist[2]
                    s3_index = stratlist[3]

                    # unpack the actual strategy values
                    s1_stratval = self.strats[s1_index]['stratval']
                    s2_stratval = self.strats[s2_index]['stratval']
                    s3_stratval = self.strats[s3_index]['stratval']

                    # this is the differential evolution "adaptive step": create a new individual
                    new_stratval = s1_stratval + self.diffevol['F'] * (s2_stratval - s3_stratval)

                    # clip to bounds
                    new_stratval = max(-1, min(+1, new_stratval))

                    # record it for future use (s0 will be evaluated first, then s_new)
                    self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval

                    if verbose:
                        print('DiffEvol: t=%.1f, s0=%d, s1=%d, (s=%+f), s2=%d, (s=%+f), s3=%d, (s=%+f), sNew=%+f' %
                              (time, self.diffevol['s0_index'],
                               s1_index, s1_stratval, s2_index, s2_stratval, s3_index, s3_stratval, new_stratval))

                    # DC's intervention for fully converged populations
                    # is the stddev of the strategies in the population equal/close to zero?
                    sum = 0.0
                    for s in range(self.k):
                        sum += self.strats[s]['stratval']
                    strat_mean = sum / self.k
                    sumsq = 0.0
                    for s in range(self.k):
                        diff = self.strats[s]['stratval'] - strat_mean
                        sumsq += (diff * diff)
                    strat_stdev = math.sqrt(sumsq / self.k)
                    if verbose:
                        print('t=,%.1f, MeanStrat=, %+f, stdev=,%f' % (time, strat_mean, strat_stdev))
                    if strat_stdev < 0.0001:
                        # this population has converged
                        # mutate one strategy at random
                        randindex = random.randint(0, self.k - 1)
                        self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)
                        if verbose:
                            print('Converged pop: set strategy %d to %+f' % (randindex, self.strats[randindex]['stratval']))

                    # set up next iteration: first evaluate s0
                    self.active_strat = self.diffevol['s0_index']
                    strat_activate(time, self.active_strat)

                    self.diffevol['de_state'] = 'active_s0'

                else:
                    sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')

        elif self.optmzr is None:
            # this is PRZI -- nonadaptive, no optimizer, nothing to change here.
            pass

        else:
            sys.exit('FAIL: bad value for self.optmzr')


class Trader_ZIP(Trader):
    initial_balance = 0
    # ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    # NB this implementation keeps separate margin values for buying & selling,
    #    so a single trader can both buy AND sell
    #    -- in the original, traders were either buyers OR sellers

    # take a ZIP strategy vector and return it as a csv-format string
    def strat_csv_str(self, strat):
        if strat is None:
            csv_str = 'None, '
        else:
            csv_str = 'mBuy=,%+5.3f, mSel=,%+5.3f, b=,%5.3f, m=,%5.3f, ca=,%6.4f, cr=,%6.4f, ' % \
                      (strat['m_buy'], strat['m_sell'], strat['beta'], strat['momntm'], strat['ca'], strat['cr'])
        return csv_str

    # how to mutate the strategy values when evolving / hill-climbing
    def mutate_strat(self, s, mode):

        def gauss_mutate_clip(value, sdev, min, max):
            mut_val = value
            while mut_val == value:
                mut_val = value + random.gauss(0.0, sdev)
                if mut_val > max:
                    mut_val = max
                elif mut_val < min:
                    mut_val = min
            return mut_val

        # mutate each element of a ZIP strategy independently
        # and clip each to remain within bounds
        if mode == 'gauss':
            big_sdev = 0.025
            small_sdev = 0.0025
            margin_buy = gauss_mutate_clip(s['m_buy'], big_sdev, -1.0, 0)
            margin_sell = gauss_mutate_clip(s['m_sell'], big_sdev, 0.0, 1.0)
            beta = gauss_mutate_clip(s['beta'], big_sdev, 0.0, 1.0)
            momntm = gauss_mutate_clip(s['momntm'], big_sdev, 0.0, 1.0)
            ca = gauss_mutate_clip(s['ca'], small_sdev, 0.0, 1.0)
            cr = gauss_mutate_clip(s['cr'], small_sdev, 0.0, 1.0)
            new_strat = {'m_buy': margin_buy, 'm_sell': margin_sell, 'beta': beta, 'momntm': momntm, 'ca': ca, 'cr': cr}
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return new_strat

    def __init__(self, ttype, tid, balance, params, time):

        Trader.__init__(self, ttype, tid, balance, params, time)

        # this set of one-liner functions named init_*() are just to make the init params obvious for ease of editing
        # for ZIP, a strategy is specified as a 6-tuple: (margin_buy, margin_sell, beta, momntm, ca, cr)
        # the 'default' values mentioned in comments below come from Cliff 1997 -- good ranges for most situations

        def init_beta():
            # in Cliff 1997 the initial beta values are U(0.1, 0.5)
            return random.uniform(0.1, 0.5)

        def init_momntm():
            # in Cliff 1997 the initial momentum values are U(0.0, 0.1)
            return random.uniform(0.0, 0.1)

        def init_ca():
            # in Cliff 1997 c_a was a system constant, the same for all traders, set to 0.05
            # here we take the liberty of introducing some variation
            return random.uniform(0.01, 0.05)

        def init_cr():
            # in Cliff 1997 c_r was a system constant, the same for all traders, set to 0.05
            # here we take the liberty of introducing some variation
            return random.uniform(0.01, 0.05)

        def init_margin():
            # in Cliff 1997 the initial margin values are U(0.05, 0.35)
            return random.uniform(0.05, 0.35)

        def init_stratwaittime():
            # not in Cliff 1997: use whatever limits you think best.
            return 7200 + random.randint(0, 3600)

        # unpack the params
        # for ZIPSH and ZIPDE params should include values for optimizer and k
        # if no params specified then defaults to ZIP with strat values as in Cliff1997

        # default parameter values
        k = 1
        optimizer = None    # no optimizer => plain non-optimizing ZIP
        logging = False

        # did call provide different params?
        if type(params) is dict:
            if 'k' in params:
                k = params['k']
            if 'optimizer' in params:
                optimizer = params['optimizer']
            self.logfile = None
            if 'logfile' in params:
                logging = True
                logfilename = params['logfile'] + '_' + tid + '_log.csv'
                self.logfile = open(logfilename, 'w')

        # the following set of variables are needed for original ZIP *and* for its optimizing extensions e.g. ZIPSH
        self.logging = logging
        self.willing = 1
        self.able = 1
        self.job = None             # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False         # gets switched to True while actively working an order
        self.prev_change = 0        # this was called last_d in Cliff'97
        self.beta = init_beta()
        self.momntm = init_momntm()
        self.ca = init_ca()         # self.ca & self.cr were hard-coded in '97 but parameterised later
        self.cr = init_cr()
        self.margin = None          # this was called profit in Cliff'97
        self.margin_buy = -1.0 * init_margin()
        self.margin_sell = init_margin()
        self.price = None
        self.limit = None
        self.prev_best_bid_p = None     # best bid price on LOB on previous update
        self.prev_best_bid_q = None     # best bid quantity on LOB on previous update
        self.prev_best_ask_p = None     # best ask price on LOB on previous update
        self.prev_best_ask_q = None     # best ask quantity on LOB on previous update

        # the following set of variables are needed only by ZIP with added hyperparameter optimization (e.g. ZIPSH)
        self.k = k                  # how many strategies evaluated at any one time?
        self.optmzr = optimizer     # what form of strategy-optimizer we're using
        self.strats = None          # the list of strategies, each of which is a dictionary
        self.strat_wait_time = init_stratwaittime()     # how many secs do we give any one strat before switching?
        self.strat_eval_time = self.k * self.strat_wait_time  # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.profit_epsilon = 0.0 * random.random()     # min profit-per-sec difference between strategies that counts

        verbose = False

        if self.optmzr is not None and k > 1:
            # we're doing some form of k-armed strategy-optimization with multiple strategies
            self.strats = []
            # strats[0] is whatever we've just assigned, and is the active strategy
            strategy = {'m_buy': self.margin_buy, 'm_sell': self.margin_sell, 'beta': self.beta,
                        'momntm': self.momntm, 'ca': self.ca, 'cr': self.cr}
            self.strats.append({'stratvec': strategy, 'start_t': time, 'active': True,
                                'profit': 0, 'pps': 0, 'evaluated': False})

            # rest of *initial* strategy set is generated from same distributions, but these are all inactive
            for s in range(1, k):
                strategy = {'m_buy': -1.0 * init_margin(), 'm_sell': init_margin(), 'beta': init_beta(),
                            'momntm': init_momntm(), 'ca': init_ca(), 'cr': init_cr()}
                self.strats.append({'stratvec': strategy, 'start_t': time, 'active': False,
                                    'profit': 0, 'pps': 0, 'evaluated': False})

        if self.logging:
            self.logfile.write('ZIP, Tid, %s, ttype, %s, optmzr, %s, strat_wait_time, %f, n_strats=%d:\n' %
                               (self.tid, self.ttype, self.optmzr, self.strat_wait_time, self.k))
            for s in self.strats:
                self.logfile.write(str(s)+'\n')

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quoteprice = int(self.limit * (1 + self.margin))

            lastprice = -1  # dummy value for if there is no lastprice
            if self.lastquote is not None:
                lastprice = self.lastquote.price

            self.price = quoteprice
            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order

            if self.logging and order.price != lastprice:
                self.logfile.write('%f, Order:, %s\n' % (time, str(order)))
        return order

    # update margin on basis of what happened in market
    def respond(self, time, lob, trade, verbose):
        # ZIP trader responds to market events, altering its margin
        # does this whether it currently has an order to work or not

        def target_up(price):
            # generate a higher target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))
            # #                        print('TargetUp: %d %d\n' % (price,target))
            return target

        def target_down(price):
            # generate a lower target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))
            # #                        print('TargetDn: %d %d\n' % (price,target))
            return target

        def willing_to_trade(price):
            # am I willing to trade at this price?
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            oldprice = self.price
            diff = price - oldprice
            change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
            self.prev_change = change
            newmargin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if newmargin < 0.0:
                    self.margin_buy = newmargin
                    self.margin = newmargin
            else:
                if newmargin > 0.0:
                    self.margin_sell = newmargin
                    self.margin = newmargin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        def load_strat(stratvec, time):
            # copy the strategy vector into the ZIP trader's params
            self.margin_buy = stratvec['m_buy']
            self.margin_sell = stratvec['m_sell']
            self.beta = stratvec['beta']
            self.momntm = stratvec['momntm']
            self.ca = stratvec['ca']
            self.cr = stratvec['cr']
            # bookkeeping
            self.n_trades = 0
            self.birthtime = time
            self.balance = 0
            self.profitpertime = 0

        def strat_activate(t, s_index):
            # print('t=%f Strat_activate, index=%d, active=%s' % (t, s_index, self.strats[s_index]['active'] ))
            self.strats[s_index]['start_t'] = t
            self.strats[s_index]['active'] = True
            self.strats[s_index]['profit'] = 0.0
            self.strats[s_index]['pps'] = 0.0
            self.strats[s_index]['evaluated'] = False

        # snapshot says whether the caller of respond() should print next frame of system snapshot data
        snapshot = False

        if self.optmzr == 'ZIPSH':

            # ZIP with simple-stochastic-hillclimber optimization of strategy (hyperparameter values)

            # NB this *cycles* through the available strats in sequence (i.e., it doesn't shuffle them)

            # first update the pps for each active strategy
            for s in self.strats:
                # update pps
                active_flag = s['active']
                if active_flag:
                    s['pps'] = self.profitpertime_update(time, s['start_t'], s['profit'])

            # have we evaluated all the strategies?
            # (could instead just compare active_strat to k, but checking them all in sequence is arguably clearer)
            # assume that all strats have been evaluated, and search for evidence to the contrary
            all_evaluated = True
            for s in self.strats:
                if s['evaluated'] is False:
                    all_evaluated = False
                    break

            if all_evaluated:
                # time to generate a new set/population of k candidate strategies
                # NB when the final strategy in the trader's set/popln is evaluated, the set is then sorted into
                # descending order of profitability, so when we get to here we know that strats[0] is elite

                if verbose and self.tid == 'S00':
                    print('t=%.3f, ZIPSH %s: strat_eval_time=%.3f,' % (time, self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('%s, start_t=%f, $=%f, pps=%f' %
                              (self.strat_csv_str(s['stratvec']), s['start_t'], s['profit'], s['pps']))

                # if the difference between the top two strats is too close to call then flip a coin
                # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                best_strat = 0
                prof_diff = self.strats[0]['pps'] - self.strats[1]['pps']
                if abs(prof_diff) < self.profit_epsilon:
                    # they're too close to call, so just flip a coin
                    best_strat = random.randint(0, 1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = self.strats[0]
                        self.strats[0] = self.strats[1]
                        self.strats[1] = tmp_strat

                # at this stage, strats[0] is our newly-chosen elite-strat, about to replicate & mutate

                # now replicate and mutate the elite into all the other strats
                for s in range(1, self.k):  # note range index starts at one not zero (elite is at [0])
                    self.strats[s]['stratvec'] = self.mutate_strat(self.strats[0]['stratvec'], 'gauss')
                    strat_activate(time, s)

                # and then update (wipe) records for the elite
                strat_activate(time, 0)

                # load the elite into the ZIP trader params
                load_strat(self.strats[0]['stratvec'], time)

                self.active_strat = 0

                if verbose and self.tid == 'S00':
                    print('%s: strat_eval_time=%f, best_strat=%d, MUTATED:' %
                          (self.tid, self.strat_eval_time, best_strat))
                    for s in self.strats:
                        print('%s start_t=%.3f, lifetime=%.3f, $=%.3f, pps=%f' %
                              (self.strat_csv_str(s['stratvec']), s['start_t'], time - s['start_t'], s['profit'],
                               s['pps']))

            else:
                # we're still evaluating

                s = self.active_strat
                time_elapsed = time - self.strats[s]['start_t']
                if time_elapsed >= self.strat_wait_time:
                    # this strategy has had long enough: update records for this strategy, then swap to another strategy
                    self.strats[s]['active'] = False
                    self.strats[s]['profit'] = self.balance
                    self.strats[s]['pps'] = self.profitpertime
                    self.strats[s]['evaluated'] = True

                    new_strat = s + 1
                    if new_strat > self.k - 1:
                        # we've just evaluated the last of this trader's set of strategies
                        # sort the strategies into order of descending profitability
                        strats_sorted = sorted(self.strats, key=lambda k: k['pps'], reverse=True)
                        # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                        # the sorted list of strats replaces the existing list
                        self.strats = strats_sorted

                        # signal that we want to record a system snapshot because this trader's eval loop finished
                        snapshot = True

                        # NB not updating self.active_strat here because next call to respond() generates new popln

                    else:
                        # copy the new strategy vector into the trader's params
                        load_strat(self.strats[new_strat]['stratvec'], time)
                        self.strats[new_strat]['start_t'] = time
                        self.active_strat = new_strat
                        self.strats[new_strat]['active'] = True
                        self.last_strat_change_time = time

                    if verbose and self.tid == 'S00':
                        if new_strat > self.k - 1:
                            print('t=%.3f (%.2fdays) %s ZIPSH respond: strat[%d] elapsed=%.3f; wait_t=%.3f, pps=%f' %
                                  (time, time / 86400, self.tid, s, time_elapsed, self.strat_wait_time, self.strats[s]['pps']))
                        else:
                            print('t=%.3f (%.2fdays) %s ZIPSH respond: strat[%d] elapsed=%.3f; wait_t=%.3f, pps=%f, switching to strat[%d]: %s' %
                                  (time, time / 86400, self.tid, s, time_elapsed, self.strat_wait_time,
                                   self.strats[s]['pps'], new_strat,
                                   self.strat_csv_str(self.strats[new_strat]['stratvec'])))

        elif self.optmzr is None:
            # this is vanilla ZIP -- nonadaptive, no optimizer, nothing to change here.
            pass

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if self.job == 'Ask':
            # seller
            if deal:
                tradeprice = trade['price']
                if self.price <= tradeprice:
                    # could sell for more? raise margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(tradeprice):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                tradeprice = trade['price']
                if self.price >= tradeprice:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(tradeprice):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        # return value of respond() tells caller whether to print a new frame of system-snapshot data
        return snapshot


class Trader_AgentPlus(Trader):

    def __init__(self, market_max_price, market_min_price=0, limit=None):
        # Parameters for aggressiveness and equilibrium price estimation
        self.theta = -5.0 * random.random()  # Risk tolerance
        self.gamma = 0.1  # Adjustment for theta based on equilibrium deviation
        self.eta = 2.0  # Price adjustment factor
        self.k = 1.0  # Risk-aversion coefficient
        self.sigma = 0.01  # Volatility estimate
        self.lambda_a = 0.01  # Aggressiveness increment
        self.lambda_r = 0.02  # Aggressiveness reduction
        self.spin_up_time = 20  # Spin-up period for market equilibrium estimation
        self.alpha = 0.1  # Smoothing parameter for EMA
        self.last_trades = []  # Track recent trades for VWAP and volatility
        self.max_trades = 10  # Limit number of trades tracked

        # Market limits
        self.market_max_price = market_max_price
        self.market_min_price = market_min_price

        # Trading state
        self.orders = []
        self.active = False
        self.limit = limit
        self.job = None  # 'Bid' or 'Ask'
        self.eqlbm = None  # Equilibrium price estimate
        self.cooldown = 0  # Cooldown timer between trades

    def update_equilibrium_price(self, price):
        """Updates the equilibrium price estimate using EMA."""
        if self.eqlbm is None:
            self.eqlbm = price
        else:
            self.eqlbm = self.alpha * price + (1 - self.alpha) * self.eqlbm

    def calculate_target(self, side):
        """Calculates target price based on limit and equilibrium."""
        if side == 'Bid':  # Buying
            if self.limit < self.eqlbm:  # Extra-marginal buyer
                target = self.limit
            else:  # Intra-marginal buyer
                target = self.eqlbm + (self.limit - self.eqlbm) * math.exp(self.k * self.sigma)
        else:  # Selling
            if self.limit > self.eqlbm:  # Extra-marginal seller
                target = self.limit
            else:  # Intra-marginal seller
                target = self.eqlbm - (self.eqlbm - self.limit) * math.exp(-self.k * self.sigma)
        return target

    def calculate_r_shout(self, target, side):
        """Calculates r_shout based on the target price."""
        if side == 'Bid':
            r_shout = math.log((target / self.eqlbm) + 1) / self.theta
        else:
            r_shout = math.log((self.eqlbm / target) + 1) / -self.theta
        return r_shout

    def update_aggressiveness(self, success_rate):
        """Adjusts aggressiveness dynamically based on trade success rate."""
        if success_rate < 0.5:  # If success rate is low, increase aggressiveness
            self.lambda_a *= 1.1
        else:  # Reduce aggressiveness slightly if success rate is high
            self.lambda_a *= 0.9

    def adjust_theta_based_on_volatility(self):
        """Adjusts risk tolerance (theta) based on market volatility."""
        if len(self.last_trades) < 2:
            return
        volatility = sum(abs(p - self.eqlbm) for p in self.last_trades) / len(self.last_trades)
        if volatility < 0.02:  # Low volatility
            self.theta *= 1.05  # Be more aggressive
        else:  # High volatility
            self.theta *= 0.95  # Be more risk-averse

    def calculate_vwap(self):
        """Calculates the Volume Weighted Average Price (VWAP)."""
        if not self.last_trades:
            return self.eqlbm
        total_volume = len(self.last_trades)  # Assuming equal volume for simplicity
        total_price = sum(self.last_trades)
        return total_price / total_volume

    def get_order(self, time, lob):
        """Generates an order based on the trader's strategy."""
        if not self.orders or self.cooldown > 0:
            self.active = False
            self.cooldown -= 1
            return None

        self.active = True
        self.limit = self.orders[0].price
        self.job = self.orders[0].otype

        # Update target price
        target = self.calculate_target(self.job)

        # Calculate quote price based on the side of the trade
        if self.job == 'Bid':  # Buying
            quote_price = min(self.limit, lob['asks']['best']) - self.eta * (lob['asks']['best'] - target)
        else:  # Selling
            quote_price = max(self.limit, lob['bids']['best']) + self.eta * (target - lob['bids']['best'])

        # Enforce minimum spread profitability
        spread = lob['asks']['best'] - lob['bids']['best']
        avg_spread = (self.market_max_price - self.market_min_price) * 0.01
        if spread < avg_spread:
            self.cooldown = 5  # Skip trading for a few ticks if the spread is too low
            return None

        # Create and return an order object
        order = {
            'type': self.job,
            'price': quote_price,
            'quantity': self.orders[0].qty,
            'time': time
        }
        return order

    def respond(self, time, lob, trade):
        """Responds to market changes and updates internal parameters."""
        if trade:
            price = trade['price']
            self.last_trades.append(price)
            if len(self.last_trades) > self.max_trades:
                self.last_trades.pop(0)  # Maintain a fixed size for the trade history
            self.update_equilibrium_price(price)
            self.theta += self.gamma * (price - self.eqlbm)  # Adjust risk tolerance dynamically
        else:
            # No trade, update theta slightly to adapt to volatility
            self.adjust_theta_based_on_volatility()

        # Adjust aggressiveness based on success rate
        success_rate = len([t for t in self.last_trades if t <= self.eqlbm]) / len(self.last_trades)
        self.update_aggressiveness(success_rate)

        # Update targets
        if self.job == 'Bid':
            target = self.calculate_target('Bid')
            self.update_aggressiveness(success_rate)
        elif self.job == 'Ask':
            target = self.calculate_target('Ask')
            self.update_aggressiveness(success_rate)

import time
import numpy as np
from scipy.special import kv  # Bessel function for long memory impact
from scipy.stats import norm


class DepthBasedTrader:
    def __init__(self, exchange_api, fill_rate_model, hawkes_model, jump_model, initial_capital=10000):
        """
        - exchange_api: API for order book data & execution.
        - fill_rate_model: Exponential decay model for execution probabilities.
        - hawkes_model: Hawkes process model for predicting order flow.
        - jump_model: Jump-diffusion model for large price movements.
        - initial_capital: Starting trading capital.
        """
        self.exchange_api = exchange_api
        self.fill_rate_model = fill_rate_model
        self.hawkes_model = hawkes_model
        self.jump_model = jump_model
        self.capital = initial_capital
        self.position = 0
        self.active_orders = {}
        self.trade_frequency = 1  # Faster reaction time (lower sleep interval)
        self.order_size_factor = 0.5  # Adjusts order size dynamically

    def get_market_data(self, symbol):
        """Fetch latest market data including depth and recent order arrivals."""
        order_book = self.exchange_api.fetch_order_book(symbol)
        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        bid_depth = sum(q for _, q in order_book['bids'][:10])
        ask_depth = sum(q for _, q in order_book['asks'][:10])
        
        # Retrieve recent market orders and cancellations
        trade_data = self.exchange_api.fetch_recent_trades(symbol)
        
        return best_bid, best_ask, bid_depth, ask_depth, trade_data

    def compute_hawkes_intensities(self, trade_data):
        """Use Hawkes model to predict future order arrival intensities."""
        timestamps = [trade['timestamp'] for trade in trade_data]
        event_types = [trade['type'] for trade in trade_data]

        # Compute updated Hawkes intensity
        predicted_intensity = self.hawkes_model.predict_intensity(timestamps, event_types)
        return predicted_intensity

    def compute_jump_probability(self, best_bid, best_ask, trade_data):
        """Estimate jump probability using a jump-diffusion model."""
        spread = best_ask - best_bid
        recent_trade_sizes = np.array([trade['size'] for trade in trade_data])

        # Estimate probability of large price move
        jump_prob = self.jump_model.compute_jump_prob(recent_trade_sizes, spread)
        return jump_prob

    def compute_optimal_order_placement(self, best_bid, best_ask, bid_depth, ask_depth, trade_data):
        """
        Determine best limit order depth using fill rates, Hawkes intensities, and jump risks.
        """
        spread = best_ask - best_bid

        # Compute fill rates using exponential decay
        lambda_minus, kappa_minus = self.fill_rate_model["buy"]
        lambda_plus, kappa_plus = self.fill_rate_model["sell"]

        buy_fill_prob = lambda_minus * np.exp(-kappa_minus * (bid_depth / bid_depth.max()))
        sell_fill_prob = lambda_plus * np.exp(-kappa_plus * (ask_depth / ask_depth.max()))

        # Predict order flow intensity using Hawkes model
        predicted_intensity = self.compute_hawkes_intensities(trade_data)

        # Compute jump risk
        jump_prob = self.compute_jump_probability(best_bid, best_ask, trade_data)

        # Define adaptive thresholds
        min_fill_threshold = 0.3
        high_jump_threshold = 0.7

        # Adjust buy order placement
        if buy_fill_prob > min_fill_threshold:
            buy_order_price = best_bid - (spread * 0.05)  # Slightly below best bid
        else:
            buy_order_price = best_bid

        # Adjust sell order placement
        if sell_fill_prob > min_fill_threshold:
            sell_order_price = best_ask + (spread * 0.05)  # Slightly above best ask
        else:
            sell_order_price = best_ask

        # Modify placement if Hawkes intensity or jump risk is high
        if predicted_intensity > 1.5:  # If order flow is highly clustered
            buy_order_price -= spread * 0.03  # More aggressive buy
            sell_order_price += spread * 0.03  # More aggressive sell
        
        if jump_prob > high_jump_threshold:  # If a jump is likely
            buy_order_price -= spread * 0.05
            sell_order_price += spread * 0.05

        return buy_order_price, sell_order_price

    def adaptive_order_size(self, predicted_intensity, jump_prob):
        """Dynamically adjust order size based on market conditions."""
        base_order_size = 1
        if predicted_intensity > 1.5:  # More orders coming in
            return base_order_size * 2  # Double order size
        if jump_prob > 0.7:  # Risk of adverse movement
            return base_order_size * 0.5  # Reduce size
        return base_order_size

    def place_orders(self, symbol):
        """Places buy and sell orders based on fill rate probabilities, Hawkes intensities, and jump risks."""
        best_bid, best_ask, bid_depth, ask_depth, trade_data = self.get_market_data(symbol)
        buy_price, sell_price = self.compute_optimal_order_placement(best_bid, best_ask, bid_depth, ask_depth, trade_data)

        # Get dynamic order size
        predicted_intensity = self.compute_hawkes_intensities(trade_data)
        jump_prob = self.compute_jump_probability(best_bid, best_ask, trade_data)
        order_size = self.adaptive_order_size(predicted_intensity, jump_prob)

        # Place limit buy order
        buy_order = self.exchange_api.create_limit_buy_order(symbol, amount=order_size, price=buy_price)
        self.active_orders[buy_order['id']] = buy_order

        # Place limit sell order
        sell_order = self.exchange_api.create_limit_sell_order(symbol, amount=order_size, price=sell_price)
        self.active_orders[sell_order['id']] = sell_order

        print(f"Placed Buy Order of {order_size} at {buy_price} and Sell Order of {order_size} at {sell_price}")

    def monitor_orders(self):
        """Monitor open orders and adjust based on market dynamics."""
        while True:
            for order_id, order in list(self.active_orders.items()):
                status = self.exchange_api.get_order_status(order_id)
                if status == "filled":
                    print(f"Order {order_id} filled.")
                    self.active_orders.pop(order_id)
                elif status in ["expired", "canceled"]:
                    print(f"Order {order_id} expired/canceled.")
                    self.active_orders.pop(order_id)

            time.sleep(self.trade_frequency)  # Adjusted frequency

    def run(self, symbol):
        """Main trading loop."""
        while True:
            self.place_orders(symbol)
            self.monitor_orders()
            time.sleep(self.trade_frequency)  # Adjust order placement dynamically

class MarketMaker:
    def __init__(self, params):
        # Parameters with defaults
        self.tick_size = params.get("tick_size", 1)                   # minimum price increment
        self.inventory_limit = params.get("inventory_limit", 20000)       # maximum absolute position allowed
        self.reequote_interval = params.get("requote_interval", 0.02)      # seconds between re-quotes
        self.order_quantity = params.get("order_quantity", 20000)         # order size
        self.levels = params.get("levels", 10)                             # number of price levels per side
        
        # We ignore balance and only track inventory
        self.position = 0  
        self.last_quote_time = None
        # Instead of single orders, we use lists for active orders
        self.active_buy_orders = []
        self.active_sell_orders = []
        # Store midprices to optionally use an older value (if desired)
        self.midprice_history = []
        # Set a profit margin so that the fill must beat the midprice by at least this much.
        self.profit_margin = params.get("profit_margin", self.tick_size)

    def update_lob(self, lob_data):
        """
        Expects lob_data as a dictionary with keys:
            "bids" and "asks", each a dict containing at least "best".
        For example, traders publish the LOB as:
          public_data = {
              "time": ...,
              "bids": {"best": best_bid, ...},
              "asks": {"best": best_ask, ...},
              "QID": ...,
              "tape": ...
          }
        """
        best_bid = lob_data.get("bids", {}).get("best", TBSE_SYS_MIN_PRICE)
        best_ask = lob_data.get("asks", {}).get("best", TBSE_SYS_MAX_PRICE)
        return best_bid, best_ask

    def get_midprice(self, best_bid, best_ask):
        """
        Calculate the midprice from the current best bid and ask.
        """
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
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
        # Optionally use an older midprice (for example, 3 events ago)
        if len(self.midprice_history) >= 3:
            effective_mid = self.midprice_history[-3]
        else:
            effective_mid = midprice

        # Build a ladder of desired buy prices.
        buy_prices = []
        for level in range(self.levels):
            # Level 0 is most aggressive: best_bid + tick_size.
            # Then step down in increments.
            price = best_bid + self.tick_size - level * self.tick_size
            # Clip so that we never buy above (midprice - profit_margin)
            price = min(price, midprice - self.profit_margin)
            buy_prices.append(int(price))
        
        # Build a ladder of desired sell prices.
        sell_prices = []
        for level in range(self.levels):
            # Level 0: best_ask - tick_size; then step up.
            price = best_ask - self.tick_size + level * self.tick_size
            # Clip so that we never sell below (midprice + profit_margin)
            price = max(price, midprice + self.profit_margin)
            sell_prices.append(int(price))

        # Safety check: if the ladders overlap, re-adjust using effective_mid.
        if max(buy_prices) >= min(sell_prices):
            buy_prices = [int(effective_mid - self.tick_size - i * self.tick_size) for i in range(self.levels)]
            sell_prices = [int(effective_mid + self.tick_size + i * self.tick_size) for i in range(self.levels)]

        # Cancel active orders that do not match desired prices.
        for order in self.active_buy_orders[:]:
            if order["price"] not in buy_prices:
                self.cancel_order(order)
                self.active_buy_orders.remove(order)
        for order in self.active_sell_orders[:]:
            if order["price"] not in sell_prices:
                self.cancel_order(order)
                self.active_sell_orders.remove(order)

        # Place new buy orders for any missing levels (if inventory allows).
        current_buy_prices = [order["price"] for order in self.active_buy_orders]
        for price in buy_prices:
            if price not in current_buy_prices and self.position > -self.inventory_limit:
                new_order = self.place_order("BUY", price, self.order_quantity)
                self.active_buy_orders.append(new_order)

        # Place new sell orders for missing levels.
        current_sell_prices = [order["price"] for order in self.active_sell_orders]
        for price in sell_prices:
            if price not in current_sell_prices and self.position < self.inventory_limit:
                new_order = self.place_order("SELL", price, self.order_quantity)
                self.active_sell_orders.append(new_order)

        self.last_quote_time = current_time

    def on_fill(self, order, fill_price):
        """
        When an order is filled, check if the trade would be losing.
        For a BUY order, we require that fill_price is less than (or equal to) our effective midprice.
        For a SELL order, fill_price must be greater than (or equal to) the effective midprice.
        If the condition is not met, we cancel the trade (i.e. we do not update our inventory).
        """
        # Use the most recent midprice as the benchmark.
        if self.midprice_history:
            current_mid = self.midprice_history[-1]
        else:
            current_mid = (TBSE_SYS_MIN_PRICE + TBSE_SYS_MAX_PRICE) / 2.0

        if order["side"] == "BUY":
            # We only want to buy at a price lower than (current_mid - profit_margin)
            if fill_price >= current_mid:
                print(f"Rejected BUY fill at {fill_price} (benchmark {current_mid}); trade would be losing.")
                self.cancel_order(order)
                return
            else:
                self.position += order["quantity"]
        elif order["side"] == "SELL":
            # We only want to sell at a price higher than (current_mid + profit_margin)
            if fill_price <= current_mid:
                print(f"Rejected SELL fill at {fill_price} (benchmark {current_mid}); trade would be losing.")
                self.cancel_order(order)
                return
            else:
                self.position -= order["quantity"]
        order["active"] = False
        print(f"Order filled: {order} at {fill_price}. New position: {self.position}")

    def run(self, lob_stream, market_order_hits):
        """
        Runs the market maker over a stream of LOB events.
        :param lob_stream: An iterable of events (dictionaries) with keys:
                           "Time": simulation time (float)
                           "LOB": dictionary containing at least "bids" and "asks" (with key "best")
        :param market_order_hits: A function that takes (order, lob_data) and returns True if that order is filled.
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
                    # Use BestAsk from lob_data as the fill price for a buy order.
                    self.on_fill(order, lob_data.get("asks", {}).get("best"))
                    self.active_buy_orders.remove(order)
            for order in self.active_sell_orders[:]:
                if market_order_hits(order, lob_data):
                    # Use BestBid from lob_data as the fill price for a sell order.
                    self.on_fill(order, lob_data.get("bids", {}).get("best"))
                    self.active_sell_orders.remove(order)


class Trader_HFTMM(Trader):
    """
    A simple high-frequency market maker trader inspired by the model in the paper.
    This trader keeps track of its inventory and uses a threshold rule:
      - If inventory is too high (above an upper threshold), it posts a sell (Ask) order.
      - If inventory is too low (below a lower threshold), it posts a buy (Bid) order.
      - Otherwise, it uses a (simulated) signal to choose its side.
    
    It then sets its quote price based on a mid-price (from the current LOB) plus an adjustment
    that grows with the absolute value of its inventory.
    """
    upper_threshold = 5
    lower_threshold = -5
    speed = 1.0
    signal_accuracy = 0.75
    balance = 10000


    def __init__(self, ttype, tid, balance, params, time):
        super().__init__(ttype, tid, balance, params, time)
        # Track net inventory (number of contracts held; positive = long, negative = short)
        self.inventory = 0  
        # Set inventory thresholds (can be passed in via params; defaults here)
        self.upper_threshold = params.get('upper_threshold', 5)
        self.lower_threshold = params.get('lower_threshold', -5)
        # Speed and signal accuracy parameters (for future use; here used in the signal simulation)
        self.speed = params.get('speed', 1.0)
        self.signal_accuracy = params.get('signal_accuracy', 0.75)
    
    def getorder(self, time, countdown, lob):
        # If no active customer order, then nothing to do
        if len(self.orders) < 1:
            return None

        # Determine order type based on current inventory:
        # - If inventory is above the upper threshold, we need to sell (post an Ask) to reduce inventory.
        # - If below the lower threshold, we need to buy (post a Bid).
        # - Otherwise, use a (simulated) signal to decide.
        if self.inventory > self.upper_threshold:
            otype = 'Ask'
        elif self.inventory < self.lower_threshold:
            otype = 'Bid'
        else:
            # Simulate a directional signal (here a simple coin flip)
            # In a more advanced implementation, you might incorporate additional state or past history.
            signal = 'Bid' if random.random() < 0.5 else 'Ask'
            otype = signal

        # Compute a mid-price from the current LOB.
        # If both bids and asks exist, take their average; otherwise fall back on a default midpoint.
        if lob['bids']['n'] > 0 and lob['asks']['n'] > 0:
            mid = (lob['bids']['best'] + lob['asks']['best']) // 2
        else:
            mid = (bse_sys_minprice + bse_sys_maxprice) // 2

        # Set an adjustment factor proportional to the absolute inventory.
        # The idea is that the more extreme the inventory, the more aggressive the pricing adjustment.
        delta = ticksize * (abs(self.inventory) + 1)
        if otype == 'Bid':
            price = max(mid - delta, bse_sys_minprice)
        else:
            price = min(mid + delta, bse_sys_maxprice)

        # Create the order using the standard Order class.
        order = Order(self.tid, otype, price, self.orders[0].qty, time, lob['QID'])
        self.lastquote = order
        return order

    def bookkeep(self, trade, order, verbose, time):
        # Use the standard bookkeeping to update blotter, balance, etc.
        super().bookkeep(trade, order, verbose, time)
        # Update inventory based on the side of the trade:
        # For a Bid order (buy), add to inventory; for an Ask order (sell), subtract.
        if order.otype == 'Bid':
            self.inventory += order.qty
        elif order.otype == 'Ask':
            self.inventory -= order.qty
        if verbose:
            print(f"Trader {self.tid} new inventory: {self.inventory}")

    def respond(self, time, lob, trade, verbose):
        # In a full implementation you could update internal parameters (e.g., thresholds or signal quality)
        # based on recent market events. For now, we leave this as a no-op.
        pass

class OFITrader:
    def __init__(self, theta_buy, theta_sell, max_wait, profit_target, stop_loss):
        self.theta_buy = theta_buy       # e.g. 0.3
        self.theta_sell = theta_sell     # e.g. 0.3
        self.max_wait = max_wait        # time steps to wait for fill
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.active_order = None
        self.position = 0
        self.entry_price = None
        self.last_timestamp = None

    def on_new_data(self, best_bid, best_ask, ofi, timestamp):
        # 1. Calculate mid-price
        mid_price = 0.5 * (best_bid + best_ask)

        # 2. If we have no active orders and no position, look for signals
        if self.active_order is None and self.position == 0:
            if ofi > self.theta_buy:
                # Place a buy limit order at best_bid
                self.active_order = {
                    "side": "BUY",
                    "price": best_bid,
                    "timestamp": timestamp
                }
            elif ofi < -self.theta_sell:
                # Place a sell limit order at best_ask
                self.active_order = {
                    "side": "SELL",
                    "price": best_ask,
                    "timestamp": timestamp
                }

        # 3. Check if active order got filled
        # (This depends on how you're simulating or connecting to the exchange.)
        # Suppose we have a function `check_fill(order, best_bid, best_ask)` that
        # returns True if the limit price is touched or crossed.
        if self.active_order is not None:
            if self.check_fill(self.active_order, best_bid, best_ask):
                # Mark position as open
                self.position = 1 if self.active_order["side"] == "BUY" else -1
                self.entry_price = self.active_order["price"]
                self.active_order = None
                self.last_timestamp = timestamp
            else:
                # If too long has passed, cancel the order
                if (timestamp - self.active_order["timestamp"]) > self.max_wait:
                    self.cancel_order(self.active_order)
                    self.active_order = None

        # 4. If we have a position, check for exit conditions
        if self.position != 0:
            # Calculate unrealized PnL
            if self.position > 0:
                # Long
                pnl = (best_bid - self.entry_price) * self.position
            else:
                # Short
                pnl = (self.entry_price - best_ask) * abs(self.position)

            # If we hit profit target or stop loss, exit
            if pnl >= self.profit_target or pnl <= -self.stop_loss:
                self.exit_position(best_bid, best_ask)
                self.position = 0
                self.entry_price = None

    def check_fill(self, order, best_bid, best_ask):
        # For a BUY order: fill if best_ask <= order["price"]
        # For a SELL order: fill if best_bid >= order["price"]
        if order["side"] == "BUY" and best_ask <= order["price"]:
            return True
        if order["side"] == "SELL" and best_bid >= order["price"]:
            return True
        return False

    def cancel_order(self, order):
        # Implementation for cancellation logic
        pass

    def exit_position(self, best_bid, best_ask):
        # Implementation for closing the position
        # e.g., place a market order or limit order at mid price
        pass


# ########################---trader-types have all been defined now--################


# #########################---Below lies the experiment/test-rig---##################


# trade_stats()
# dump CSV statistics on exchange data and trader population to file for later analysis
# this makes no assumptions about the number of types of traders, or
# the number of traders of any one type -- allows either/both to change
# between successive calls, but that does make it inefficient as it has to
# re-analyse the entire set of traders on each call
def trade_stats(expid, traders, dumpfile, time, lob):

    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %06d, ' % (expid, time))

    # second two columns of output are the LOB best bid and best offer (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

    dumpfile.write('\n')


# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
def populate_market(traders_spec, traders, shuffle, verbose):
    # traders_spec is a list of buyer-specs and a list of seller-specs
    # each spec is (<trader type>, <number of this type of trader>, optionally: <params for this type of trader>)

    def trader_type(robottype, name, parameters):
        balance = 0.00
        time0 = 0
        if robottype == 'GVWY':
            return Trader_Giveaway('GVWY', name, balance, parameters, time0)
        elif robottype == 'ZIC':
            return Trader_ZIC('ZIC', name, balance, parameters, time0)
        elif robottype == 'SHVR':
            return Trader_Shaver('SHVR', name, balance, parameters, time0)
        elif robottype == 'SNPR':
            return Trader_Sniper('SNPR', name, balance, parameters, time0)
        elif robottype == 'ZIP':
            return Trader_ZIP('ZIP', name, balance, parameters, time0)
        elif robottype == 'ZIPSH':
            return Trader_ZIP('ZIPSH', name, balance, parameters, time0)
        elif robottype == 'PRZI':
            return Trader_PRZI('PRZI', name, balance, parameters, time0)
        elif robottype == 'PRSH':
            return Trader_PRZI('PRSH', name, balance, parameters, time0)
        elif robottype == 'PRDE':
            return Trader_PRZI('PRDE', name, balance, parameters, time0)
        elif robottype == 'AgentPlus':
            return Trader_AgentPlus('AgentPlus', name, balance, parameters, time0)
        elif robottype == 'Seller':
            return Trader_Seller('Seller',name, balance, parameters, time0)
        elif robottype == 'MarketMaker':
            return Trader_Human('MarketMaker',name, balance, parameters, time0)
        elif robottype == 'OFITrader':
            return OFITrader('OFITrader',name, balance, parameters, time0)
        elif robottype == 'HFTMM':
            return Trader_HFTMM('HFTMM',name, balance, parameters, time0)
        else:
            sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

    def shuffle_traders(ttype_char, n, traders):
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            traders[t1name].tid = t2name
            traders[t2name].tid = t1name
            temp = traders[t1name]
            traders[t1name] = traders[t2name]
            traders[t2name] = temp

    def unpack_params(trader_params, mapping, ttype):
        # Default parameters to avoid NoneType errors
        default_params = {'optimizer': None, 'k': 1, 'strat_min': -1.0, 'strat_max': 1.0}

        # Unpack parameters for each trader type
        if ttype in ['ZIPSH', 'ZIP']:
            if mapping:
                return 'landscape-mapper'
            elif trader_params is not None:
                parameters = trader_params.copy()
                parameters['optimizer'] = 'ZIPSH' if ttype == 'ZIPSH' else None
                return parameters
            else:
                return default_params  # Use default if None

        elif ttype in ['PRSH', 'PRDE', 'PRZI']:
            if mapping:
                return 'landscape-mapper'
            elif trader_params is not None:
                return {
                    'optimizer': 'PRSH' if ttype == 'PRSH' else 'PRDE' if ttype == 'PRDE' else None,
                    'k': trader_params.get('k', default_params['k']),
                    'strat_min': trader_params.get('s_min', default_params['strat_min']),
                    'strat_max': trader_params.get('s_max', default_params['strat_max']),
                }
            else:
                return default_params  # Use default if None

        return default_params  # Ensures function always returns valid parameters


    landscape_mapping = False   # set to true when mapping fitness landscape (for PRSH etc).

    # the code that follows is a bit of a kludge, needs tidying up.
    n_buyers = 0
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            if len(bs) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(bs[2], landscape_mapping, ttype)
            else:
                params = unpack_params(None, landscape_mapping, ttype)
            traders[tname] = trader_type(ttype, tname, params)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            if len(ss) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(ss[2], landscape_mapping, ttype)
            else:
                params = unpack_params(None, landscape_mapping, ttype)
            traders[tname] = trader_type(ttype, tname, params)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)

    if verbose:
        for t in range(n_buyers):
            bname = 'B%02d' % t
            print(traders[bname])
        for t in range(n_sellers):
            bname = 'S%02d' % t
            print(traders[bname])

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers}


# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring


import sys, random, numpy as np
import time as chrono


import sys, random, numpy as np, time as chrono

# Global trade history – this should be updated in market_session as trades occur.
global_trade_history = []


import random
import numpy as np

def weibull_kernel(delta, alpha, shape, scale):
    """
    Weibull kernel for delta = t - t_i > 0.
    Returns: α * (shape/scale) * ((delta)/scale)^(shape-1) * exp[-((delta)/scale)^shape]
    """
    if delta < 0:
        return 0.0
    return alpha * (shape / scale) * ((delta) / scale)**(shape - 1) * math.exp(-((delta) / scale)**shape)

def get_dynamic_issuetimes_hawkes(n_traders, current_time, trade_history,
                                  base_rate=0.2, alpha=1.2,
                                  kernel='weibull', weibull_shape=1.5, weibull_scale=1.2, beta = 0.6, delta = -1):
    times = []       # List to store accepted waiting times (relative to current_time)
    t = current_time
    # Start with a copy of past events so that new events affect future intensity
    events = list(trade_history)
    count = 0

    while count < n_traders:
        # Calculate current intensity lambda(t)
        if kernel == 'weibull':
            intensity_contrib = sum(weibull_kernel(t - ti, alpha, weibull_shape, weibull_scale)
                                    for ti in events if t >= ti)
            lambda_t = base_rate + intensity_contrib
        else:
            raise ValueError(f"Kernel {kernel} not supported.")

        # Draw a candidate waiting time from an exponential with mean 1/lambda_t
        waiting_time = np.random.exponential(1.0 / lambda_t)
        t_candidate = t + waiting_time

        # Calculate intensity at candidate time t_candidate
        if kernel == 'weibull':
            intensity_candidate = sum(weibull_kernel(t_candidate - ti, alpha, weibull_shape, weibull_scale)
                                      for ti in events if t_candidate >= ti)
            lambda_candidate = base_rate + intensity_candidate

        # Thinning: accept candidate with probability lambda_candidate / lambda_t
        if random.random() <= lambda_candidate / lambda_t:
            t = t_candidate
            events.append(t)  # Record new event so it affects future intensity
            times.append(t - current_time)
            count += 1
        else:
            t = t_candidate  # Advance time even if not accepted

    return np.array(times)

# --- Modified customer_orders() function ---
def customer_orders(time, last_update, traders, trader_stats, os, pending, verbose):
    def sysmin_check(price):
        if price < bse_sys_minprice:
            print('WARNING: price < bse_sys_min -- clipped')
            price = bse_sys_minprice
        return price

    def sysmax_check(price):
        if price > bse_sys_maxprice:
            print('WARNING: price > bse_sys_max -- clipped')
            price = bse_sys_maxprice
        return price

    def getorderprice(i, sched, n, mode, issuetime):
        if len(sched[0]) > 2:
            offsetfn = sched[0][2]
            if callable(offsetfn):
                offset_min = offsetfn(issuetime)
                offset_max = offset_min
            else:
                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
            if len(sched[0]) > 3:
                offsetfn = sched[0][3]
                if callable(offsetfn):
                    offset_max = offsetfn(issuetime)
                else:
                    sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
        else:
            offset_min = 0.0
            offset_max = 0.0

        pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
        pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
        prange = pmax - pmin
        stepsize = prange / (n - 1)
        halfstep = round(stepsize / 2.0)

        if mode == 'fixed':
            orderprice = pmin + int(i * stepsize)
        elif mode == 'jittered':
            orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
        elif mode == 'random':
            if len(sched) > 1:
                s = random.randint(0, len(sched) - 1)
                pmin = sysmin_check(min(sched[s][0], sched[s][1]))
                pmax = sysmax_check(max(sched[s][0], sched[s][1]))
            orderprice = random.randint(int(pmin), int(pmax))
        elif mode == 'custom':
            # CUSTOM MODE: Calculate a factor based on the movement of the last 10 trades.
            # Here we assume a global list 'recent_trades' exists, which holds the prices of the last 10 trades.
            if 'recent_trades' in globals() and len(recent_trades) >= 1:
                # Calculate movement: difference between most recent and the oldest of the 10 trades.
                movement = recent_trades[-1] - recent_trades[-5]
                # Compute a factor: for example, let factor = 1 + (movement / prange)
                # Clamp the factor to a reasonable range to avoid too extreme step sizes.
                factor = max(0.2, min(1.5, 3.3 + (movement / prange)))
            else:
                factor = 1.0  # Fallback if not enough data
            custom_stepsize = stepsize * factor
            orderprice = pmin + int(i * custom_stepsize)
        else:
            sys.exit('FAIL: Unknown mode in schedule')
        orderprice = sysmin_check(sysmax_check(orderprice))
        return orderprice

    def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in getissuetimes()')
        
        if mode == 'weibull':
            # Use a Weibull distribution to generate interarrival times.
            # Choose a shape parameter not equal to 1. For instance, shape < 1 produces clustering.
            shape = 0.5  # Adjust this value to get more or less clustering.
            # np.random.weibull returns samples from a Weibull distribution with the given shape.
            interarrival_times = np.random.weibull(a=shape, size=n_traders)
            # Optionally, scale the total time to match the desired interval:
            total_time = sum(interarrival_times)
            scale_factor = interval / total_time
            interarrival_times = [t * scale_factor for t in interarrival_times]
            issuance_times = np.cumsum(interarrival_times)
            if shuffle:
                issuance_times = list(issuance_times)
                random.shuffle(issuance_times)
            return issuance_times
        if mode == 'weibel':
            return get_dynamic_issuetimes_hawkes(n_traders, current_time, trade_history,
                                                   base_rate=0.7, alpha=0.2,
                                                   kernel='weibull', weibull_shape=1.5, weibull_scale=1.0)
        # Check for our custom non-homogeneous mode:
        if mode == 'nonhomogeneous':
            # Use the global_trade_history (assumed to be maintained in market_session)
            #return get_dynamic_issuetimes(n_traders, interval, time, global_trade_history, window=15, base_rate=10)
            return get_dynamic_issuetimes_hawkes(n_traders, time, global_trade_history, base_rate=0.5, alpha=0.2, beta=1.0)
        # Otherwise, use the original modes.
        elif mode == 'periodic':
            # For periodic mode, simply generate equally spaced times.
            tstep = interval / n_traders
            return [t * tstep for t in range(n_traders)]
        elif mode == 'drip-fixed':
            tstep = interval / (n_traders - 1) if n_traders > 1 else interval
            return [t * tstep for t in range(n_traders)]
        elif mode == 'drip-jitter':
            tstep = interval / (n_traders - 1) if n_traders > 1 else interval
            return [t * tstep + tstep * random.random() for t in range(n_traders)]
        elif mode == 'drip-poisson':
            arrtime = 0
            times = []
            for _ in range(n_traders):
                arrtime += random.expovariate(n_traders / interval)
                times.append(arrtime)
            return times
        else:
            sys.exit('FAIL: unknown time-mode in getissuetimes()')

    def getschedmode(time, os):
        got_one = False
        schedrange = None
        mode = None
        for sched in os:
            if (sched['from'] <= time) and (time < sched['to']):
                schedrange = sched['ranges']
                mode = sched['stepmode']
                got_one = True
                break
        if not got_one:
            sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
        return schedrange, mode

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']
    shuffle_times = True
    cancellations = []

    if len(pending) < 1:
        new_pending = []
        # Buyers:
        issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)
        ordertype = 'Bid'
        (sched, mode) = getschedmode(time, os['dem'])
        for t in range(n_buyers):
            issuetime = time + issuetimes[t]
            tname = 'B%02d' % t
            orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)
        # Sellers:
        issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
        ordertype = 'Ask'
        (sched, mode) = getschedmode(time, os['sup'])
        for t in range(n_sellers):
            issuetime = time + issuetimes[t]
            tname = 'S%02d' % t
            orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)
    else:
        new_pending = []
        for order in pending:
            if order.time < time:
                tname = order.tid
                response = traders[tname].add_order(order, verbose)
                if verbose:
                    print('Customer order: %s %s' % (response, order))
                if response == 'LOB_Cancel':
                    cancellations.append(tname)
            else:
                new_pending.append(order)
    return [new_pending, cancellations]

# one session in the market
def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, dump_flags, verbose):

    tape_dump_interval = (60*5)  # 5 minutes
    last_tape_dump_time = starttime
    def dump_strats_frame(time, stratfile, trdrs):
        # write one frame of strategy snapshot

        line_str = 't=,%.0f, ' % time

        best_buyer_id = None
        best_buyer_prof = 0
        best_buyer_strat = None
        best_seller_id = None
        best_seller_prof = 0
        best_seller_strat = None

        # loop through traders to find the best
        for t in traders:
            trader = trdrs[t]

            # print('PRSH/PRDE/ZIPSH strategy recording, t=%s' % trader)
            if trader.ttype == 'PRSH' or trader.ttype == 'PRDE' or trader.ttype == 'ZIPSH':
                line_str += 'id=,%s, %s,' % (trader.tid, trader.ttype)

                if trader.ttype == 'ZIPSH':
                    # we know that ZIPSH sorts the set of strats into best-first
                    act_strat = trader.strats[0]['stratvec']
                    act_prof = trader.strats[0]['pps']
                else:
                    act_strat = trader.strats[trader.active_strat]['stratval']
                    act_prof = trader.strats[trader.active_strat]['pps']

                line_str += 'actvstrat=,%s ' % trader.strat_csv_str(act_strat)
                line_str += 'actvprof=,%f, ' % act_prof

                if trader.tid[:1] == 'B':
                    # this trader is a buyer
                    if best_buyer_id is None or act_prof > best_buyer_prof:
                        best_buyer_id = trader.tid
                        best_buyer_strat = act_strat
                        best_buyer_prof = act_prof
                elif trader.tid[:1] == 'S':
                    # this trader is a seller
                    if best_seller_id is None or act_prof > best_seller_prof:
                        best_seller_id = trader.tid
                        best_seller_strat = act_strat
                        best_seller_prof = act_prof
                else:
                    # wtf?
                    sys.exit('unknown trader id type in market_session')

        if best_buyer_id is not None:
            line_str += 'best_B_id=,%s, best_B_prof=,%f, best_B_strat=, ' % (best_buyer_id, best_buyer_prof)
            line_str += traders[best_buyer_id].strat_csv_str(best_buyer_strat)

        if best_seller_id is not None:
            line_str += 'best_S_id=,%s, best_S_prof=,%f, best_S_strat=, ' % (best_seller_id, best_seller_prof)
            line_str += traders[best_seller_id].strat_csv_str(best_seller_strat)

        line_str += '\n'

        if verbose:
            print('line_str: %s' % line_str)
        stratfile.write(line_str)
        stratfile.flush()
        os.fsync(stratfile)

    def blotter_dump(session_id, traders):
        bdump = open(session_id+'_blotters.csv', 'w')
        for t in traders:
            bdump.write('%s, %d\n' % (traders[t].tid, len(traders[t].blotter)))
            for b in traders[t].blotter:
                bdump.write('%s, %s, %.3f, %d, %s, %s, %d\n'
                            % (traders[t].tid, b['type'], b['time'], b['price'], b['party1'], b['party2'], b['qty']))
        bdump.close()

    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False

    if dump_flags['dump_strats']:
        strat_dump = open(sess_id + '_strats.csv', 'w')
    else:
        strat_dump = None

    if dump_flags['dump_lobs']:
        lobframes = open(sess_id + '_LOB_frames.csv', 'w')
    else:
        lobframes = None

    if dump_flags['dump_avgbals']:
        avg_bals = open(sess_id + '_avg_balance.csv', 'w')
    else:
        avg_bals = None

    # initialise the exchange
    exchange = Exchange()

    # create a bunch of traders
    traders = {}
    trader_stats = populate_market(trader_spec, traders, True, populate_verbose)

    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

    duration = float(endtime - starttime)

    last_update = -1.0

    time = starttime

    pending_cust_orders = []

    if verbose:
        print('\n%s;  ' % sess_id)

    # frames_done is record of what frames we have printed data for thus far
    frames_done = set()

    while time < endtime:

        # how much time left, as a percentage?
        time_left = (endtime - time) / duration

        # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

        trade = None

        [pending_cust_orders, kills] = customer_orders(time, last_update, traders, trader_stats,
                                                       order_schedule, pending_cust_orders, orders_verbose)

        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].lastquote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    exchange.del_order(time, traders[kill].lastquote, verbose)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
        order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lobframes, lob_verbose))

        # if verbose: print('Trader Quote: %s' % (order))

        if order is not None:
            if order.otype == 'Ask' and order.price < traders[tid].orders[0].price:
                sys.exit('Bad ask')
            if order.otype == 'Bid' and order.price > traders[tid].orders[0].price:
                sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade = exchange.process_order2(time, order, process_verbose)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters
                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
                if dump_flags['dump_avgbals']:
                    trade_stats(sess_id, traders, avg_bals, time, exchange.publish_lob(time, lobframes, lob_verbose))

            # traders respond to whatever happened
            lob = exchange.publish_lob(time, lobframes, lob_verbose)
            any_record_frame = False
            for t in traders:
                # NB respond just updates trader's internal variables
                # doesn't alter the LOB, so processing each trader in
                # sequence (rather than random/shuffle) isn't a problem
                record_frame = traders[t].respond(time, lob, trade, respond_verbose)
                if record_frame:
                    any_record_frame = True

            # log all the PRSH/PRDE/ZIPSH strategy info for this timestep?
            if any_record_frame and dump_flags['dump_strats']:
                # print one more frame to strategy dumpfile
                dump_strats_frame(time, strat_dump, traders)
                # record that we've written this frame
                frames_done.add(int(time))

        time = time + timestep

    # session has ended

    # write trade_stats for this session (NB could use this to write end-of-session summary only)
    if dump_flags['dump_avgbals']:
        trade_stats(sess_id, traders, avg_bals, time, exchange.publish_lob(time, lobframes, lob_verbose))
        avg_bals.close()

    if dump_flags['dump_tape']:
        # dump the tape (transactions only -- not writing cancellations)
        exchange.tape_dump(sess_id + '_tape.csv', 'w', 'keep')

    if dump_flags['dump_blotters']:
        # record the blotter for each trader
        blotter_dump(sess_id, traders)

    if dump_flags['dump_strats']:
        strat_dump.close()

    if dump_flags['dump_lobs']:
        lobframes.close()


    
def market_session_event_driven(sess_id, starttime, endtime, trader_spec, order_schedule, dump_flags, verbose):
    exchange = Exchange()
    traders = {}
    trader_stats = populate_market(trader_spec, traders, True, verbose=False)

    time = starttime
    pending_cust_orders = []
    # get initial set of orders
    pending_cust_orders, kills = customer_orders(time, None, traders, trader_stats, order_schedule, pending_cust_orders, False)

    while True:
        # 1) If no pending orders, or all are beyond endtime, we are done
        future_orders = [o for o in pending_cust_orders if o.time > time]
        if len(future_orders) == 0:
            break

        # 2) Find the earliest order
        next_order = min(future_orders, key=lambda o: o.time)
        next_time = next_order.time

        if next_time > endtime:
            break

        # 3) Advance simulation time to that event time
        time = next_time

        # 4) Issue all orders whose time <= current time
        newly_due = [o for o in pending_cust_orders if o.time <= time]
        still_pending = [o for o in pending_cust_orders if o.time > time]
        for order in newly_due:
            tname = order.tid
            # Add the order to that trader
            response = traders[tname].add_order(order, verbose=False)
            if response == 'LOB_Cancel' and traders[tname].lastquote is not None:
                exchange.del_order(time, traders[tname].lastquote, verbose=False)
        pending_cust_orders = still_pending

        # 5) Possibly pick a random trader to quote, etc.
        # or you can skip this step if you only want arrivals at these event times
        if len(traders) > 0:
            tid = random.choice(list(traders.keys()))
            lob = exchange.publish_lob(time, None, False)
            order = traders[tid].getorder(time, endtime - time, lob)
            if order is not None:
                trade = exchange.process_order2(time, order, False)
                if trade is not None:
                    # update trader blotters
                    traders[trade['party1']].bookkeep(trade, order, False, time)
                    traders[trade['party2']].bookkeep(trade, order, False, time)

        # 6) Possibly fetch new orders from customer_orders() if it schedules repeated cycles
        # e.g. if the schedule says we re-generate at intervals. You might call:
        pending_cust_orders, kills = customer_orders(time, None, traders, trader_stats, order_schedule, pending_cust_orders, False)
        # remove kills, etc.

        # repeat

    # end of session
    # dump tapes, stats, etc. if needed
    exchange.tape_dump(sess_id + '_tape.csv', 'w', 'keep')



