class Trader_PRZI(Trader):

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
