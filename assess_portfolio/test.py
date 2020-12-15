"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'], \
                     allocs=[0.1, 0.2, 0.3, 0.4], \
                     sv=1000000, rfr=0.0, sf=252.0, \
                     gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    #plot_data(prices_all)
    prices_all = fill_missing_values(prices_all)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later


    # Get daily portfolio value
    # step1 - normed : normed=prices/prices[0] - only portfolio symbols
    normed = normalize_data(prices)
    # step2 - alloced : alloces=mormed*allocs
    alloced = normed * allocs
    # step3 - pos_vals :  pos_vals=alloced*start_val
    pos_vals = alloced * sv
    # step4 - port_val
    port_val = pos_vals.copy().sum(axis=1)

    # port_val = prices_SPY # add code here to compute daily portfolio values

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Portfolio Daily returns
    daily_ret = compute_daily_returns(port_val)

    # Cumulative return
    cr = (port_val[-1] / port_val[0]) - 1

    # Average period return (if sf == 252 this is daily return)
    adr = daily_ret.mean()

    # Standard deviation of daily return
    sddr = daily_ret.std()

    # Sharpe ratio
    sf = np.int_(sf)
    k = np.sqrt(sf)
    daily_rf=float(1+rfr)** (1/float(252)) - 1
    print "Alla --- k ", k
    print "Alla --- rfr", rfr
    print "Alla --- ",daily_rf
   # sr = k * (adr - daily_rf) / sddr
    sr = k * np.mean(daily_ret - daily_rf) / sddr


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        normed_temp = normalize_data(df_temp)
        plot_data(normed_temp)
        pass

    # Add code here to properly compute end value

    # Calculate End value of portfolio
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)
    return df_data

# Normalize data
def normalize_data(df):
    return df / df.ix[0]


# Compute and return the daily return values
def compute_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0] = 0  # set daily returns for row 0 to 0
    return daily_returns[1:]


#Ploting data
def plot_data(df_data,title='Daily portfolio value and SPY',xlable="Date",ylable="Normalized price"):
    ax = df_data.plot(title=title)
    ax.set_xlabel(xlable)
    ax.set_ylabel(ylable)
    # output a plot such as plot.png
    plt.savefig('plot.png')
    plt.show()


def  run_test(start_date, end_date, symbols, allocations, start_val, risk_free_rate, sample_freq):
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!



    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd=start_date, ed=end_date, \
                                             syms=symbols, \
                                             allocs=allocations, \
                                             sv=start_val, \
                                             gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print "End Value", ev

    print "---------------------------------------------"

def  test_code():
    #run_test(dt.datetime(2010,1,1), dt.datetime(2010,12,31),['GOOG', 'AAPL', 'GLD', 'XOM'],[0.2, 0.3, 0.4, 0.1],1000000,0.0,252)
    #run_test(dt.datetime(2010,1,1), dt.datetime(2010, 12, 31),['AXP', 'HPQ', 'IBM', 'HNZ'],[0.0, 0.0, 0.0, 1.0],1000000,0.0,252)
    #run_test(dt.datetime(2010,6,1), dt.datetime(2010,12,31),['GOOG', 'AAPL', 'GLD', 'XOM'],[0.2, 0.3, 0.4, 0.1],1000000,0.0,252)

    #additional test cases
    #run_test(dt.datetime(2010,9,1), dt.datetime(2010,12,31),['IYR'],[1.0],1000000,0.1,252)
   run_test(dt.datetime(2009,7,2), dt.datetime(2010,7,30),['USB','VAR'],[0.3, 0.7],1000000,0.02,252)
   # run_test(dt.datetime(2008,6,3), dt.datetime(2010,6,29),['HSY','VLO','HOT'],[0.2, 0.4, 0.4],1000000,0.03,252)
   # run_test(dt.datetime(2007,5,4), dt.datetime(2010,5,28),['VNO','WU','EMC','AMGN'],[0.2, 0.3, 0.4, 0.1],1000000,0.04,252)
   # run_test(dt.datetime(2005,4,6), dt.datetime(2010,3,25),['ETN','KSS','NYT','GPS','BMC','TEL'],[0.2, 0.1, 0.1, 0.1, 0.4, 0.1],1000000,0.06,252)
   # run_test(dt.datetime(2003,2,8), dt.datetime(2010,1,23),['HRL','SDS','ACS','IFF','WMB','FFIV','BK','AIV'],[0.2, 0.2, 0.1, 0.1, 0.2, 0, 0.2, 0],1000000,0.08,252)
   # run_test(dt.datetime(2002,2,9), dt.datetime(2010,10,22),['CCT','JNJ','ERTS','MCO','R','WDC','BBT','JOY','PLL'],[0.2, 0.2, 0.1, 0.1, 0.2, 0, 0, 0.2, 0],1000000,0.09,252)


if __name__ == "__main__":
    test_code()
