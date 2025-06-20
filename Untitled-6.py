

    def getissuetimes(n_traders, market_volatility, interval, shuffle=False, fittointerval=True):
        # Validate that n_traders is at least 1.
        market_volatility = 0.2
        if n_traders < 1:
            raise ValueError("n_traders must be at least 1")
        
        # Ensure market_volatility is numeric.
        try:
            vol = float(market_volatility)
        except ValueError:
            raise ValueError("market_volatility must be a numeric value")
        
        interval = float(interval)
        
        # Determine the baseline time step.
        if n_traders == 1:
            base_tstep = interval
        else:
            base_tstep = interval / (n_traders - 1)
        
        # Adjust the time step based on market volatility.
        # With higher volatility, we reduce the time step, so orders come in faster.
        tstep = base_tstep / (1 + vol)
        
        # Generate the issuance times deterministically.
        issuetimes = [t * tstep for t in range(n_traders)]
        
        # If requested, scale the times so that the last issuance time exactly equals 'interval'.
        if fittointerval:
            last_time = issuetimes[-1]
            if last_time != 0:
                scale_factor = interval / last_time
                issuetimes = [time * scale_factor for time in issuetimes]
        
        # Optionally, shuffle the order of issuance times.
        if shuffle:
            import random
            random.shuffle(issuetimes)
        
        return issuetimes
