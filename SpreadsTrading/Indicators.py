import pandas as pd
from pandas import Series, DataFrame, TimeSeries
import numpy as np
from datetime import datetime, timedelta
import time
import os
import math
from sympy import Symbol, sqrt
from sympy.solvers import nsolve

def rsi(closes, period=14):
    num_closes = len(closes)    
    if num_closes < period:
        print ("rsi Error: Check data frame length")
        raise SystemExit
    # this could be named gains/losses to save time/memory in the future    
    changes = closes[1:] - closes[:-1]
    #num_changes = len(changes)
    length_diff = period
    rsi_range = num_closes - length_diff
    rsis = np.zeros(rsi_range)
    avg_gain = np.zeros(rsi_range)
    avg_loss = np.zeros(rsi_range)
    gains = np.array(changes)
    # assign 0 to all negative values
    masked_gains = gains < 0
    gains[masked_gains] = 0
    #print gains
    losses = np.array(changes)
    # assign 0 to all positive values
    masked_losses = losses > 0
    losses[masked_losses] = 0
    losses *= -1
    avg_gain[0] = np.sum(gains[:period-1]) / period
    avg_loss[0] = np.sum(losses[:period-1]) / period
    if avg_loss[0] == 0:
        rsis[0] = 100
    else:
        rs = avg_gain[0] / avg_loss[0]
        rsis[0] = 100 - (100 / (1 + rs))
    for idx in range(1, rsi_range):
        avg_gain[idx] = (avg_gain[idx-1] * (period - 1) + gains[idx + (period - 1)]) / period
        avg_loss[idx] = (avg_loss[idx-1] * (period - 1) + losses[idx + (period - 1)]) / period
        if avg_loss[idx] == 0:
            rsis[idx] = 100
        else:
            rs = avg_gain[idx] / avg_loss[idx]
            rsis[idx] = 100 - (100 / (1 + rs))
    return (rsis, avg_gain, avg_loss)

def moving_average(list_price, n):
    l = [0] + list_price
    arr = np.cumsum(l, dtype=float)
    return (arr[n:] - arr[:-n]) / n

def sma(closes, period):
    num_closes = len(closes)
    if num_closes < period:
        print ("sma Error: Check data frame length")
        raise SystemExit
    length_diff = period - 1
    sma_range = num_closes - length_diff
    smas = np.zeros(sma_range)    
    for idx in range(sma_range):
        smas[idx] = np.mean(closes[idx:idx + period])
    return smas

def ema(closes, period):
    num_closes = len(closes)
    if num_closes < period:
        raise SystemExit    
    ema_range = num_closes - period + 1   
    emas = np.zeros(ema_range)       
    emas[0] = np.mean(closes[:period])
    smooth_const = 2 / float(period + 1)    
    for idx in range(1, ema_range):
        emas[idx] = smooth_const * (closes[period + idx - 1] - emas[idx - 1]) + emas[idx - 1]
    return emas    

def atr(opens, closes, period):
	num_closes = len(closes)
	if num_closes < period:
		print ("atr Error: Check data frame length")
		raise SystemExit
	s1 = abs(opens[1:] - closes[:-1])
	s1 = np.append(np.array([0]), s1)
	s2 = abs(closes[1:] - closes[:-1])
	s2 = np.append(np.array([0]), s2)
	s3 = abs(opens - closes)
	true_range = np.zeros(num_closes)
	for i in range(len(true_range)):
		true_range[i] = max(s1[i], s2[i], s3[i])
	return sma(true_range, period)	
	#return ema(true_range, period)

def bb(closes, period, up_std_dev, low_std_dev):
    num_closes = len(closes)
    if num_closes < period:
        print ("bb Error: Check data frame length")
        raise SystemExit
    bb_range = num_closes - period + 1
    # 3 bands, bandwidth, range and %B
    bbs_upper = np.zeros(bb_range)
    bbs_mid = np.zeros(bb_range)
    bbs_lower = np.zeros(bb_range)
    simpleMA = sma(closes, period)
    for idx in range(bb_range):
        std_dev = np.std(closes[idx:idx + period])
        #print std_dev
        # upper, middle, lower bands and the bandwidth
        bbs_upper[idx] = simpleMA[idx] + std_dev * up_std_dev
        bbs_mid[idx] = simpleMA[idx]
        bbs_lower[idx] = simpleMA[idx] - std_dev * low_std_dev
    return (bbs_upper, bbs_mid, bbs_lower)