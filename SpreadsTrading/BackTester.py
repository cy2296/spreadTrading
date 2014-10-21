import pandas as pd
from pandas import Series, DataFrame, TimeSeries
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter, WeekdayLocator, MonthLocator, HourLocator, DayLocator, MONDAY
from matplotlib.finance import plot_day_summary
import matplotlib.dates as mdates
import os
import math
##############################################################################################################
##############################################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""" FUNCTIONS FOR COLLECTING SPREAD DATA """""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################################################################################
##############################################################################################################
def stringDateTime_toDatetime(string):
    return datetime.strptime(string, "%Y%m%d %H:%M")

def stringDate_toDatetime(string):
    return datetime.strptime(string, "%Y%m%d")

def datetime_toString(dt):
    return dt.strftime("%Y%m%d")


def convertFormat(m, d):
	if m < 10: mm = '0' + str(m)
	else: mm = str(m)
	if d < 10: dd = '0' + str(d)    
	else: dd = str(d)
	return mm + dd

def month_toNumber(str_month):
	if str_month == 'F': return 1
	elif str_month == 'G': return 2
	elif str_month == 'H': return 3
	elif str_month == 'J': return 4
	elif str_month == 'K': return 5 
	elif str_month == 'M': return 6
	elif str_month == 'N': return 7
	elif str_month == 'Q': return 8
	elif str_month == 'U': return 9
	elif str_month == 'V': return 10
	elif str_month == 'X': return 11
	elif str_month == 'Z': return 12
#findYearCase('LE', 4, 8, 12, 3, 2003)
def findYearCase(symbol, m_buy, m_sell, m_in, m_out, year):
	if symbol == 'CL' or symbol == 'NG' or symbol == 'XRB':
		terminate_m_buy = m_buy	- 1
		terminate_m_sell = m_sell - 1
	else:
		terminate_m_buy = m_buy
		terminate_m_sell = m_sell
	# entry month <= exit month
	if m_in <= m_out:
		# both contracts doesn't expire during the trading period     
		if terminate_m_buy >= m_out and terminate_m_sell >= m_out: 
			y_in = y_out = y_buy = y_sell = year	
		# sell contract expire during the period, so sell next-year contract
		elif terminate_m_buy >= m_out and terminate_m_sell < m_out:
			y_in = y_out = y_buy = year
			y_sell = year + 1		
		# buy contract expire during the period, so buy next-year contract
		elif terminate_m_buy < m_out and terminate_m_sell >= m_out:
			y_in = y_out = y_sell = year
			y_buy = year + 1
		# both ontracts expire during the period, so buy & sell next-year contract
		else:
			y_in = y_out = year
			y_buy = y_sell = year + 1
	else:   
		if terminate_m_buy >= m_out and terminate_m_sell >= m_out:
			y_in = year
			y_out = y_buy = y_sell = year + 1
		elif terminate_m_buy >= m_out and terminate_m_sell < m_out:
			y_in = year
			y_out = y_in = year + 1
			y_sell = year + 2
		elif terminate_m_buy < m_out and terminate_m_sell >= m_out:
			y_in = year
			y_out = y_sell = year + 1
			y_buy = year + 2
		else:
			y_in = year
			y_out = year + 1
			y_buy = y_sell = year + 2
	return (y_buy, y_sell, y_in, y_out)

""" Input: dataFrame, entryDate, exitDate """
def findDataRange (data, str_Colum, dt_entryDate, dt_exitDate, days_before, days_after):
	# find the start date 
	#d_out = dt_exitDate
	if len(data) == 0:
		print ("findDataRange Error 1: no data during trading period")
		dataBegin = entry = exit = dataEnd = 0
	else:
		for i in range(len(data)):
			d1 = stringDate_toDatetime(str(data.ix[i, str_Colum]))
			if d1 >= dt_entryDate:
				#print d1		#debug
				if d1 >= dt_exitDate:
					print ("findDataRange Error 2: no data during trading period")
					dataBegin = entry = exit = dataEnd = 0	
					break 
				else: 
					entry = i
					if i - days_before >= 0:
						dataBegin = i - days_before
					else:
						dataBegin = 0
					break
		if i == len(data) - 1:
			print ("findDataRange Error 3: no data during trading period")
			dataBegin = entry = exit = dataEnd = len(data)
		for j in range(entry, len(data)):
			d2 = stringDate_toDatetime(str(data.ix[j, str_Colum]))
			#print d2
			if d2 > dt_exitDate:
				exit = j - 1
				#print "exit:"
				#print exit
				if exit + days_after <= len(data) - 1:
					dataEnd = exit + days_after
					break
				else:
					dataEnd = len(data) - 1
					break
			""" if the last data is still < dt_exitDate """		
			if j == len(data) - 1 and d2 <= dt_exitDate:
				exit = dataEnd = len(data) - 1		
	#print (dataBegin, entry, exit, dataEnd)		
	return (dataBegin, entry, exit, dataEnd)

##############################################################################################################
##############################################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""" FUNCTIONS FOR TECHNICAL INDICATORS """""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################################################################################
##############################################################################################################
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
    gains = np.array(changes)
    # assign 0 to all negative values
    masked_gains = gains < 0
    gains[masked_gains] = 0
    losses = np.array(changes)
    # assign 0 to all positive values
    masked_losses = losses > 0
    losses[masked_losses] = 0
    losses *= -1
    # convert all negatives into positives    losses *= -1
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsis[0] = 100
    else:
        rs = avg_gain / avg_loss
        rsis[0] = 100 - (100 / (1 + rs))

    for idx in range(1, rsi_range):
        avg_gain = (avg_gain * (period - 1) + gains[idx + (period - 1)]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx + (period - 1)]) / period
        if avg_loss == 0:
            rsis[idx] = 100
        else:
            rs = avg_gain / avg_loss
            rsis[idx] = 100 - (100 / (1 + rs))
    return rsis

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
		# upper, middle, lower bands and the bandwidth
		bbs_upper[idx] = simpleMA[idx] + std_dev * up_std_dev
		bbs_mid[idx] = simpleMA[idx]
		bbs_lower[idx] = simpleMA[idx] - std_dev * low_std_dev
	return (bbs_upper, bbs_mid, bbs_lower)



##############################################################################################################
##############################################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""		
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""" FUNCTIONS FOR GENERATING TRADING SIGNALS """""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################################################################################
##############################################################################################################
def diff(arr1, arr2):
    assert (len(arr1) >= len(arr2))
    arr1 = arr1[len(arr1) - len(arr2):] 
    diff = arr1 - arr2
    return diff

# find the i where arr[i-1] = 0 and arr[i] = arr_val, then assign arr[i] = assign_val
def findCrossover(arr, arr_val, c_up_val, c_dwn_val):
	arr_val = abs(arr_val)
	changes = arr[1:] - arr[:-1]
	changes = numpy.append(np.array([0]), changes)
	for i in range(len(changes)):
		if changes[i] == arr_val: # cross up
			arr[i] = 0
			changes[i] = c_up_val
		elif changes[i] == -arr_val:
			arr[i] = 0
			changes[i] = c_dwn_val
	return arr + changes

def initStringList(string, len):
	s_list = []
	for i in range(len):
		s_list.append(string)
	return s_list

def pnl_toDollarValue(symbol, pnl):
	if symbol == 'ZS' or symbol == 'ZC' or symbol == 'ZW' or symbol == 'KW':
		dollarPNL = pnl * 50
	elif symbol == 'ZM':
		dollarPNL = pnl * 100 
	elif symbol == 'HG':
		dollarPNL = pnl * 250
	elif symbol == 'LE' or symbol == 'LH':
		dollarPNL = pnl * 400
	elif symbol == 'GF':
		dollarPNL = pnl * 500
	elif symbol == 'ZL':
		dollarPNL = pnl * 600
	elif symbol == 'CL':
		dollarPNL = pnl * 1000
	elif symbol == 'NG':
		dollarPNL = pnl * 10000
	elif symbol == 'HO' or symbol == 'XRB':	
		dollarPNL = pnl * 42000
	return dollarPNL

def dollarPNL_toPNL(symbol, dollarPNL):
	if symbol == 'ZS' or symbol == 'ZC' or symbol == 'ZW' or symbol == 'KW':
		pnl = float(dollarPNL) / 50
	elif symbol == 'ZM':
		pnl = float(dollarPNL) / 100
	elif symbol == 'HG':
		pnl = float(dollarPNL) / 250
	elif symbol == 'LE' or symbol == 'LH':
		pnl = float(dollarPNL) / 400
	elif symbol == 'GF':
		pnl = float(dollarPNL) / 500
	elif symbol == 'ZL':
		pnl = float(dollarPNL) / 600
	elif symbol == 'CL':
		pnl = float(dollarPNL) / 1000
	elif symbol == 'NG':
		pnl = float(dollarPNL) / 10000
	elif symbol == 'HO' or symbol == 'XRB':	
		pnl = float(dollarPNL) / 42000
	return pnl

""" simply hold the position during the trading period """
def strat_simple_hold(idx, closes, symbol, hard_stop, atr_stop, tradePeriod, rsi_event, rsi_event_id, \
	action, posSize, posPrc, dynamicPNL, realizedPNL):
	hard_stop *= -1
	c_blw_l = rsi_event_id[0]
	blw_l = rsi_event_id[1]
	c_abv_l = rsi_event_id[2]
	in_band = rsi_event_id[3]
	c_abv_h = rsi_event_id[4]
	abv_h = rsi_event_id[5]
	c_blw_h = rsi_event_id[6]
	if tradePeriod[k] == 1:
		if posSize[k-1] == 0:
			if rsi_event[k] == c_abv_l:
				action[k] = 'long'
				posSize[k] = 1
				posPrc[k] = closes[k]
		#''' has a position, keep updating information'''
		elif posSize[k-1] == 1:
			dollarAtrStop = pnl_toDollarValue(symbol, 2 * atr_stop[k-1])		
			stopLoss = min(hard_stop, dollarAtrStop) 
			pnl = closes[k] - posPrc[k-1]
			dollarPNL = abs(pnl_toDollarValue(symbol, pnl))
			if pnl < 0: 
				if dollarPNL >= stopLoss and stopLoss == dollarAtrStop:
					action[k] = 'atr stop long'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] - 2 * atr_stop[k-1]
					realizedPNL[k] = -1 * 2 * atr_stop[k-1]
				elif dollarPNL >= stopLoss and stopLoss == hard_stop:	
					action[k] = 'hard stop long'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] - 2 * dollarPNL_toPNL(symbol, hard_stop)
					realizedPNL[k] = -1 * dollarPNL_toPNL(symbol, hard_stop)
				else:
					action[k] = 'keep long'
					posSize[k] = 1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = closes[k] - posPrc[k]
			elif pnl >= 0:
				#''' exit long '''
				if rsi_event[k] == c_abv_h:
					action[k] = 'exit long'
					posSize[k] = 0  
					#action[k] = 'Trailing stop'
					#posSize[k] = 1  # realize profit and keep position
					posPrc[k] = closes[k]
					realizedPNL[k] = closes[k] - posPrc[k-1]
				else:
					action[k] = 'keep long'
					posSize[k] = 1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = closes[k] - posPrc[k]
	elif tradePeriod[k] == 0:
		#'''exit pos at close trading date'''
		if posSize[k-1] == 1:
			action[k] = 'close long'
			posSize[k] = 0
			posPrc[k] = closes[k]
			realizedPNL[k] = closes[k] - posPrc[k-1]

""" Long-only rsi crossover bb with min(atrStop, hardStop) """
def strat_rsi_bb_crossover_1(idx, closes, symbol, hard_stop, atr_stop, tradePeriod, rsi_event, rsi_event_id, \
	action, posSize, posPrc, dynamicPNL, realizedPNL):
	hard_stop *= -1
	c_blw_l = rsi_event_id[0]
	blw_l = rsi_event_id[1]
	c_abv_l = rsi_event_id[2]
	in_band = rsi_event_id[3]
	c_abv_h = rsi_event_id[4]
	abv_h = rsi_event_id[5]
	c_blw_h = rsi_event_id[6]
	if tradePeriod[k] == 1:
		if posSize[k-1] == 0:
			if rsi_event[k] == c_abv_l:
				action[k] = 'long'
				posSize[k] = 1
				posPrc[k] = closes[k]
		#''' has a position, keep updating information'''
		elif posSize[k-1] == 1:
			dollarAtrStop = pnl_toDollarValue(symbol, 2 * atr_stop[k-1])		
			stopLoss = min(hard_stop, dollarAtrStop) 
			pnl = closes[k] - posPrc[k-1]
			dollarPNL = abs(pnl_toDollarValue(symbol, pnl))
			if pnl < 0: 
				if dollarPNL >= stopLoss and stopLoss == dollarAtrStop:
					action[k] = 'atr stop long'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] - 2 * atr_stop[k-1]
					realizedPNL[k] = -1 * 2 * atr_stop[k-1]
				elif dollarPNL >= stopLoss and stopLoss == hard_stop:	
					action[k] = 'hard stop long'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] - 2 * dollarPNL_toPNL(symbol, hard_stop)
					realizedPNL[k] = -1 * dollarPNL_toPNL(symbol, hard_stop)
				else:
					action[k] = 'keep long'
					posSize[k] = 1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = closes[k] - posPrc[k]
			elif pnl >= 0:
				#''' exit long '''
				if rsi_event[k] == c_abv_h:
					action[k] = 'exit long'
					posSize[k] = 0  
					#action[k] = 'Trailing stop'
					#posSize[k] = 1  # realize profit and keep position
					posPrc[k] = closes[k]
					realizedPNL[k] = closes[k] - posPrc[k-1]
				else:
					action[k] = 'keep long'
					posSize[k] = 1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = closes[k] - posPrc[k]
	elif tradePeriod[k] == 0:
		#'''exit pos at close trading date'''
		if posSize[k-1] == 1:
			action[k] = 'close long'
			posSize[k] = 0
			posPrc[k] = closes[k]
			realizedPNL[k] = closes[k] - posPrc[k-1]

""" Short-only rsi crossover bb with min(atrStop, hardStop) """
def strat_rsi_bb_crossover_2(idx, closes, symbol, hard_stop, atr_stop, tradePeriod, rsi_event, rsi_event_id, \
	action, posSize, posPrc, dynamicPNL, realizedPNL):
	hard_stop *= -1
	c_blw_l = rsi_event_id[0]
	blw_l = rsi_event_id[1]
	c_abv_l = rsi_event_id[2]
	in_band = rsi_event_id[3]
	c_abv_h = rsi_event_id[4]
	abv_h = rsi_event_id[5]
	c_blw_h = rsi_event_id[6]
	if tradePeriod[k] == 1:
		if posSize[k-1] == 0:
			if rsi_event[k] == c_blw_h:
				action[k] = 'short'
				posSize[k] = -1
				posPrc[k] = closes[k]
		elif posSize[k-1] == -1:
			dollarAtrStop = pnl_toDollarValue(symbol, 2 * atr_stop[k-1])		
			stopLoss = min(hard_stop, dollarAtrStop) 
			pnl = posPrc[k-1] - closes[k] 
			dollarPNL = abs(pnl_toDollarValue(symbol, pnl))
			if pnl < 0: 
				if dollarPNL >= stopLoss and stopLoss == dollarAtrStop:
					action[k] = 'atr stop short'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] + 2 * atr_stop[k-1]
					realizedPNL[k] = -1 * 2 * atr_stop[k-1]
				elif dollarPNL >= stopLoss and stopLoss == hard_stop:	
					action[k] = 'hard stop short'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] + 2 * dollarPNL_toPNL(symbol, hard_stop)
					realizedPNL[k] = -1 * dollarPNL_toPNL(symbol, hard_stop)
				else:
					action[k] = 'keep short'
					posSize[k] = -1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = posPrc[k] - closes[k] 
			elif pnl >= 0:
				#''' exit long '''
				if rsi_event[k] == c_blw_l:
					action[k] = 'exit short'
					posSize[k] = 0  
					#action[k] = 'Trailing stop'
					#posSize[k] = 1  # realize profit and keep position
					posPrc[k] = closes[k]
					realizedPNL[k] =  posPrc[k-1] - closes[k]
				else:
					action[k] = 'keep short'
					posSize[k] = -1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = posPrc[k] - closes[k] 
	elif tradePeriod[k] == 0:
		#'''exit pos at close trading date'''
		if posSize[k-1] == -1:
			posSize[k] = 0
			posPrc[k] = closes[k]
			action[k] = 'close short'
			realizedPNL[k] = posPrc[k-1] - closes[k] 			

""" Both direction rsi crossover bb with min(atrStop, hardStop) """
def strat_rsi_bb_crossover_3(idx, closes, symbol, hard_stop, atr_stop, tradePeriod, rsi_event, rsi_event_id, \
	action, posSize, posPrc, dynamicPNL, realizedPNL):
	hard_stop *= -1
	c_blw_l = rsi_event_id[0]
	blw_l = rsi_event_id[1]
	c_abv_l = rsi_event_id[2]
	in_band = rsi_event_id[3]
	c_abv_h = rsi_event_id[4]
	abv_h = rsi_event_id[5]
	c_blw_h = rsi_event_id[6]
	if tradePeriod[k] == 1:
		if posSize[k-1] == 0:
			if rsi_event[k] == c_abv_l:
				action[k] = 'long'
				posSize[k] = 1
				posPrc[k] = closes[k]
			elif rsi_event[k] == c_blw_h:
				action[k] = 'short'
				posSize[k] = -1
				posPrc[k] = closes[k]
		#''' has a position, keep updating information'''
		elif posSize[k-1] == 1:
			dollarAtrStop = pnl_toDollarValue(symbol, 2 * atr_stop[k-1])		
			stopLoss = min(hard_stop, dollarAtrStop) 
			pnl = closes[k] - posPrc[k-1]
			dollarPNL = abs(pnl_toDollarValue(symbol, pnl))
			if pnl < 0: 
				if dollarPNL >= stopLoss and stopLoss == dollarAtrStop:
					action[k] = 'atr stop long'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] - 2 * atr_stop[k-1]
					realizedPNL[k] = -1 * 2 * atr_stop[k-1]
				elif dollarPNL >= stopLoss and stopLoss == hard_stop:	
					action[k] = 'hard stop long'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] - 2 * dollarPNL_toPNL(symbol, hard_stop)
					realizedPNL[k] = -1 * dollarPNL_toPNL(symbol, hard_stop)
				else:
					action[k] = 'keep long'
					posSize[k] = 1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = closes[k] - posPrc[k]
			elif pnl >= 0:
				#''' exit long '''
				if rsi_event[k] == c_abv_h:
					action[k] = 'exit long'
					posSize[k] = 0  
					#action[k] = 'Trailing stop'
					#posSize[k] = 1  # realize profit and keep position
					posPrc[k] = closes[k]
					realizedPNL[k] = closes[k] - posPrc[k-1]
				else:
					action[k] = 'keep long'
					posSize[k] = 1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = closes[k] - posPrc[k]
		elif posSize[k-1] == -1:
			dollarAtrStop = pnl_toDollarValue(symbol, 2 * atr_stop[k-1])		
			stopLoss = min(hard_stop, dollarAtrStop) 
			pnl = posPrc[k-1] - closes[k] 
			dollarPNL = abs(pnl_toDollarValue(symbol, pnl))
			if pnl < 0: 
				if dollarPNL >= stopLoss and stopLoss == dollarAtrStop:
					action[k] = 'atr stop short'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] + 2 * atr_stop[k-1]
					realizedPNL[k] = -1 * 2 * atr_stop[k-1]
				elif dollarPNL >= stopLoss and stopLoss == hard_stop:	
					action[k] = 'hard stop short'
					posSize[k] = 0
					posPrc[k] = posPrc[k-1] + 2 * dollarPNL_toPNL(symbol, hard_stop)
					realizedPNL[k] = -1 * dollarPNL_toPNL(symbol, hard_stop)
				else:
					action[k] = 'keep short'
					posSize[k] = -1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = posPrc[k] - closes[k] 
			elif pnl >= 0:
				#''' exit long '''
				if rsi_event[k] == c_blw_l:
					action[k] = 'exit short'
					posSize[k] = 0  
					#action[k] = 'Trailing stop'
					#posSize[k] = 1  # realize profit and keep position
					posPrc[k] = closes[k]
					realizedPNL[k] =  posPrc[k-1] - closes[k]
				else:
					action[k] = 'keep short'
					posSize[k] = -1
					posPrc[k] = posPrc[k-1]
					dynamicPNL[k] = posPrc[k] - closes[k] 
	elif tradePeriod[k] == 0:
		#'''exit pos at close trading date'''
		if posSize[k-1] != 0:
			posSize[k] = 0
			posPrc[k] = closes[k]
			if posSize[k-1] == 1:
				action[k] = 'close long'
				realizedPNL[k] = closes[k] - posPrc[k-1]
			elif posSize[k-1] == -1:
				action[k] = 'close short'
				realizedPNL[k] = posPrc[k-1] - closes[k] 
##############################################################################################################
##############################################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""		
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""" FUNCTIONS FOR MANAGING PROFIT AND LOSS """""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################################################################################
##############################################################################################################
def addDollarPNLColumn(symbol, df_result, str_col, str_new_col):
	if symbol == 'ZS' or symbol == 'ZC' or symbol == 'ZW' or symbol == 'KW':
		df_result[str_new_col] = df_result[str_col] * 50
	elif symbol == 'ZM':
		df_result[str_new_col] = df_result[str_col] * 100
	elif symbol == 'HG':	
		df_result[str_new_col] = df_result[str_col] * 25000
	elif symbol == 'LE' or symbol == 'LH':
		df_result[str_new_col] = df_result[str_col] * 400
	elif symbol == 'GF':
		df_result[str_new_col] = df_result[str_col] * 500
	elif symbol == 'ZL':
		df_result[str_new_col] = df_result[str_col] * 600
	elif symbol == 'CL':		
		df_result[str_new_col] = df_result[str_col] * 1000
	elif symbol == 'NG':
		df_result[str_new_col] = df_result[str_col] * 10000
	elif symbol == 'HO' or symbol == 'XRB':	
		df_result[str_new_col] = df_result[str_col] * 42000
	return df_result

def addDailyPNLChange(df, DynamicDollarPNL, RealizedDollarPNL, DailyPNLChange):
	totalPNL = df['DynamicDollarPNL'] + df['RealizedDollarPNL']
	totalPNL = np.asarray(totalPNL)
	arr1 = totalPNL[:-1]
	arr2 = totalPNL[1:]
	change = arr2 - arr1
	change = np.append(np.array([0]), change)
	df['DailyPNLChange'] = change
	return tradeSummary

# input: a list of trade results, trade id
# output: a dataFrame that contains the report for trade id
def mergeResultList(resultList):
	mergeResult = resultList[0]
	if len(resultList) > 1:
		for i in range(1, len(resultList)):
			mergeResult = mergeResult.append(resultList[i], ignore_index = True)
			#result['Trades'] = title
	return mergeResult

def getDateCode(dt_series):
	dt_code = []
	for i in range(len(dt_series)):
		dt_code.append(date2num(dt_series[i]))
	return dt_code	

def getDateInt(dt_series):
	dt_int = []
	for i in range(len(dt_series)):
		dt_int.append(int(datetime_toString(dt_series[i])))
	return dt_int

def flag_fitTimeFrame(dfList, colName, timeFrame, timeFrameIdx):
	dataList = []
	for i in range(len(dfList)):
		tf = timeFrame
		df = dfList[i].set_index(timeFrameIdx)
		tf[colName] = df[colName]		
		tf = tf.fillna(method = 'ffill')
		tf = tf.fillna(0)
		dataList.append(np.asarray(tf[colName]))
	return dataList

def pnl_fitTimeFrame(dfList, colName, timeFrame, timeFrameIdx):
	dataList = []
	for i in range(len(dfList)):
		tf = timeFrame
		df = dfList[i].set_index(timeFrameIdx)
		tf[colName] = df[colName]		
		tf = tf.fillna(0)
		dataList.append(np.asarray(tf[colName]))
	return dataList	

def mergeDataList(dataList):
	totalData = dataList[0]	
	for i in range(1, len(dataList)):
		totalData += dataList[i]
	return totalData

def set_timeFrameIndex(dfList, timeFrameIdx):
	for i in range(len(dfList)):
		dfList[i] = dfList[i].set_index(timeFrameIdx)
	return dfList

def seriesData_fitTimeFrame(dfList, colName, ts):
	dataList = []
	#ts = TimeSeries(pd.DateRange(datetime(1998,1,1), datetime(2013, 3, 20)))
	for i in range(len(dfList)):
		tf = DataFrame(index = ts)
		df = dfList[i]
		tf[colName] = df[colName]
		if colName == 'TradeID' or colName == 'Symbol' or colName == 'Year' or colName == 'Buy' or colName == 'Sell' \
			or colName == 'Strat' or colName == 'StartTrading' or colName == 'EndTrading':	
			tf = tf.fillna(method = 'bfill')
			tf = tf.fillna(-99)
		elif colName == 'Close' or colName == 'IsPeriod' or colName == 'PosSize' or colName == 'PosDir' \
			or colName == 'DynamicPNL' or colName == 'DynamicDollarPNL':
			tf = tf.fillna(method = 'ffill')
			tf = tf.fillna(0)
		elif colName == 'RealizedPNL' or colName == 'RealizedDollarPNL' or colName == 'DailyPNLChange':
			tf = tf.fillna(0)
		dataList.append(np.asarray(tf[colName]))
	return dataList	


def seriesAction_fitTimeFrame(dfList, Action, ts, PosDirList):
	dataList = []
	for i in range(len(dfList)):
		tf = DataFrame(index = ts)
		df = dfList[i]
		tf[Action] = df[Action]
		tf = tf.fillna(-99)
		tf.reset_index(inplace = True)
		tf['PosDir'] = PosDirList[i]
		for j in range(len(tf)):
			if tf.ix[j, Action] == -99:
				if j == 0: 
					tf.ix[j, Action] = 'none'
				elif tf.ix[j-1, 'PosDir'] == 1:
					tf.ix[j, Action] = 'keep long'
				elif tf.ix[j-1, 'PosDir'] == -1:
					tf.ix[j, Action] = 'keep short'	
				else:
					tf.ix[j, Action] = 'none'
		dataList.append(np.asarray(tf[Action]))
	return dataList	
				

def seriesPosPrc_fitTimeFrame(dfList, PosPrc, ts, PosDirList):
	dataList = []
	for i in range(len(dfList)):
		tf = DataFrame(index = ts)
		df = dfList[i]
		tf[PosPrc] = df[PosPrc]
		tf = tf.fillna(-99999)
		tf.reset_index(inplace = True)
		tf['PosDir'] = PosDirList[i]
		for j in range(len(tf)):
			if tf.ix[j, PosPrc] == -99999:
				if j == 0: tf.ix[j, PosPrc] = 0
				elif tf.ix[j-1, 'PosDir'] != 0:
					tf.ix[j, PosPrc] = tf.ix[j-1, PosPrc]
				else:
					tf.ix[j, PosPrc] = 0
		dataList.append(np.asarray(tf[PosPrc]))
	return dataList	

def tradeSummary_fitTimeFrame(trade, dfList, ts, TradeID, Year, Symbol, StartTrading, EndTrading, Date, Close, IsPeriod, Action, \
					PosSize, PosDir, PosPrc, DynamicPNL, RealizedPNL, DynamicDollarPNL, RealizedDollarPNL, DailyPNLChange):
	tf = DataFrame(index = ts)
	df = dfList[trade]
	tf[TradeID] = df[TradeID]
	tf[Year] = df[Year]
	tf[Symbol] = df[Symbol]
	#tf[Buy] = df[Buy]
	#tf[Sell] = df[Sell]
	tf[StartTrading] = df[StartTrading]
	tf[EndTrading] = df[EndTrading]
	tf[Date] = df[Date]
	tf[Close] = df[Close]
	tf[IsPeriod] = df[IsPeriod]
	tf[Action] = df[Action]
	tf[PosSize] = df[PosSize]
	tf[PosDir] = df[PosDir]
	tf[PosPrc] = df[PosPrc]
	tf[DynamicPNL] = df[DynamicPNL]
	tf[RealizedPNL] = df[RealizedPNL]
	tf[DynamicDollarPNL] = df[DynamicDollarPNL]
	tf[RealizedDollarPNL] = df[RealizedDollarPNL]
	tf[DailyPNLChange] = df[DailyPNLChange]
	return tf	

def tradeSummary_fillMissing(x, ts, TradeIDList, YearList, SymbolList, StartTradingList, \
					EndTradingList, t_int, CloseList, IsPeriodList, ActionList, PosSizeList, \
					PosDirList, PosPrcList, DynamicPNLList, RealizedPNLList, DynamicDollarPNLList, RealizedDollarPNLList, DailyPNLChangeList):
	tf = DataFrame(index = ts)
	tf['TradeID'] = TradeIDList[x]
	tf['Year'] = YearList[x]
	tf['Symbol'] = SymbolList[x]
	#tf['Buy'] = BuyList[x]
	#tf['Sell'] = SellList[x]
	tf['StartTrading'] = StartTradingList[x]
	tf['EndTrading'] = EndTradingList[x]
	tf['Date'] = t_int
	tf['Close'] = CloseList[x]
	tf['IsPeriod'] = IsPeriodList[x]
	tf['Action'] = ActionList[x]
	tf['PosSize'] = PosSizeList[x]
	tf['PosDir'] = PosDirList[x]
	tf['PosPrc'] = PosPrcList[x]
	tf['DynamicPNL'] = DynamicPNLList[x]
	tf['RealizedPNL'] = RealizedPNLList[x]
	tf['DynamicDollarPNL'] = DynamicDollarPNLList[x]
	tf['RealizedDollarPNL'] = RealizedDollarPNLList[x]
	tf['DailyPNLChange'] = DailyPNLChangeList[x]
	return tf

def findEndTradingDate(symbol, year, month):
	if symbol == 'ZL' or symbol == 'ZS' or symbol == 'ZM' or symbol == 'ZC' or symbol == 'ZW' or symbol == 'GF' or symbol == 'LH': 
		date = 10
		end_trading_date = str(year) + convertFormat(month, date)
	elif symbol == 'CL' or symbol == 'HO' or symbol == 'NG' or symbol == 'XRB' or symbol =='LE':
		date = 20
		if month == 1:
			year -= 1
			end_trading_date = str(year) + convertFormat(12, date)
		else:
			end_trading_date = str(year) + convertFormat(month-1, date)
	elif symbol == 'HG':
		date = 5
		end_trading_date = str(year) + convertFormat(month, date)
	return stringDate_toDatetime(end_trading_date)	


##############################################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""" SETTING THE STRATEGY PARAMETERS  """""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################################################################################
##############################################################################################################
tradeStart = 0
tradeEnd = 48
backTestYearStart = 2001
backTestYearEnd = 2014
curr_year = 2014

allTime = 1
follow_strat = 0
days_before_expire = 90
early_transfer = 8
################## parameters ####################
days_before = 30  # for defining seasonal window
days_after = 0
_rsi_len = 10
_atr_len = 15
_bb_len = 15
_up_bb = 1.95
_dwn_bb = 1.95
_stoploss = 0
_hard_stop = -2000
##############################################################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""  THE MAIN ALGORITHM START  """""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##############################################################################################################
##############################################################################################################

data = pd.read_csv('C:\\Users\\GYANG\\Google Drive\\Historical Data\\Sensonal_Fu_D\\MRCI.csv')
MRCI = DataFrame(data)
os.chdir ("C:\\Users\\GYANG\\Google Drive\\Historical Data\\Sensonal_Fu_D\\Single_Months_D\\data")

tradeSummaryList = []
for i in range(tradeStart,tradeEnd+1):
	tradeID = i
	tradeTitle = "Trade#" + str(i) + " Buy" + " " + MRCI.ix[i, 'Buy'] + " " + MRCI.ix[i, 'b_Month'] + " " + \
					"Sell" + " "+ MRCI.ix[i, 'Sell'] + " " + MRCI.ix[i, 's_Month']
	print ("Title: " + tradeTitle)
	#startTradingList = []
	#endTradingList = []
	buy_symbol = MRCI.ix[i, 'Buy']
	m_buy = month_toNumber(MRCI.ix[i, 'b_Month'])
	sell_symbol = MRCI.ix[i, 'Sell']
	m_sell = month_toNumber(MRCI.ix[i, 's_Month'])
	m_in = MRCI.ix[i, 'i_Month']
	m_out = MRCI.ix[i, 'o_Month']

	#numYears = MRCI.ix[i, 'endYear'] - MRCI.ix[i, 'startYear'] + 1
	numYears = backTestYearEnd - backTestYearStart + 1

	strat = MRCI.ix[i, 'action']
	yearSummaryList = []		# The report for a sigle trade report

	for j in range(backTestYearStart, backTestYearEnd+1):
		year = j
		print ("year: " + str(year))
		#### check red year ####
		(y_buy, y_sell, y_in, y_out) = findYearCase(buy_symbol, m_buy, m_sell, m_in, m_out, year)
		if MRCI.ix[i, 's_red'] == 1:
			y_sell = y_buy + 1

		#end_trading_date = findEndTradingDate(buy_symbol, y_buy, m_buy)
		#start_trading_date = end_trading_date - timedelta(days_before_expire)
		#print ("endDate: " + str(end_trading_date))

		if y_in < curr_year:
			case = 'historical'	## what if end_trading_date is in this year ? 
		elif y_in == curr_year:
			if start_trading_date <= datetime.now() and datetime.now() <= end_trading_date: 
				case = 'live'
			elif datetime.now() > end_trading_date:	
				case = 'historical'
			else:
				print ('no')
				break	
		else: # y_in > curr_year	
			if start_trading_date <= datetime.now():
				case = 'live'
			else:
				print ('no')
				break

		############ find the files ############
		#if case == 'live':
		#	symbol = buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy)[3] + ":" + sell_symbol + MRCI.ix[i, 's_Month'] + str(y_sell)[3]
		#	f = buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy)[3] + "~" + sell_symbol + MRCI.ix[i, 's_Month'] + str(y_sell)[3] + "_D.csv"
		#	print (f)
		#	data = pd.read_csv(f)
		#	df = DataFrame(data)
		#else: 
		symbol = MRCI.ix[i, 'Buy'] + " " + MRCI.ix[i, 'b_Month'] + str(y_buy) + " - " \
			+ MRCI.ix[i, 'Sell'] + " " + MRCI.ix[i, 's_Month'] + str(y_sell)
		''' find out the file '''
		f1 = MRCI.ix[i, 'Buy'] + " " + MRCI.ix[i, 'b_Month'] + str(y_buy) + "_D.csv"
		f2 = MRCI.ix[i, 'Sell'] + " " + MRCI.ix[i, 's_Month'] + str(y_sell) + "_D.csv"
		#if os.path.exists(f1) != True:
		#	f1 = MRCI.ix[i, 'Buy'] + " " + MRCI.ix[i, 'b_Month'] + str(y_buy)[3] + "_D.csv"	
		#if os.path.exists(f2) != True:
		#	f2 = MRCI.ix[i, 'Sell'] + " " + MRCI.ix[i, 's_Month'] + str(y_sell)[3] + "_D.csv"
		print ("BUY: " + f1)
		print ("SELL: " + f2)
		#print "BUY: " + buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy)
		#print "SELL: " + sell_symbol + " " + MRCI.ix[i, 's_Month'] + str(y_sell)
		data1 = pd.read_csv(f1)
		data2 = pd.read_csv(f2)
		buy = DataFrame(data1)
		sell = DataFrame(data2)
		########## Get spread dataFrame ############
		buy = buy.set_index('Date')
		sell = sell.set_index('Date')
		df = buy - sell		
		df = df.dropna()
		df.Volume[df.Volume >= 0] = sell.Volume 	# set the volume
		df.Volume[df.Volume < 0] = buy.Volume	# set the volume
		df.reset_index(inplace = True)  # reset index for findDataRange
		############ find data range ############
		least_effective_length = _rsi_len + _bb_len

		if len(df) > least_effective_length:	
			if allTime == 1:
				if j == backTestYearStart: 
					end_trading_date = stringDate_toDatetime(str(df.ix[len(df)-early_transfer, 'Date']))
					start_trading_date = end_trading_date - timedelta(365)
					print (year, start_trading_date, end_trading_date)
				else: 
					start_trading_date = end_trading_date + timedelta(1)
					end_trading_date = stringDate_toDatetime(str(df.ix[len(df)-early_transfer, 'Date']))
					print (year, start_trading_date, end_trading_date)
			elif allTime == 0:
				end_trading_date = stringDate_toDatetime(str(df.ix[len(df)-early_transfer, 'Date']))
				start_trading_date = end_trading_date - timedelta(days_before_expire)

			(idx_beg, idx_beg, idx_end, idx_end) = findDataRange(df, 'Date', start_trading_date, end_trading_date, 0, 0)
			#print (idx_beg, idx_beg, idx_end, idx_end)
			df = df[idx_beg:idx_end+1]
			df.reset_index(inplace = True)
			idx_beg = 0
			idx_end = len(df) - 1
			#print "len df: ", len(df) 
			#print "idx_beg:", idx_beg
			#print "idx_end:", idx_end
			########################################################################################	
			if len(df) > least_effective_length:
				t = []
				dt = []
				for t_idx in range(len(df)):
					d = stringDate_toDatetime(str(df.ix[t_idx, 'Date']))
					dt.append(d) # debug
					t.append(date2num(d))
				#################### Trading Strategies #############################################
				arr_closes = np.asarray(df.Close)
				arr_opens = np.asarray(df.Open)
				RSI = rsi(arr_closes, _rsi_len)
				(BB_h, BB_m, BB_l) = bb(RSI, _bb_len, _up_bb, _dwn_bb)
				avg_trueRange = atr(arr_opens, arr_closes, _atr_len)
				#########################################################################################
				avg_trueRange = np.append(np.array([avg_trueRange[0] for x in range(_atr_len-1)]), avg_trueRange)
				""""""""""""""""""""""""""""""""" RSI EVENT ARRAY """""""""""""""""""""""""""""""""""""""
				''' RSI Event ID ''' 
				'''(c_blw_l, blw_l, c_abv_l, in_band, c_abv_h, abv_h, c_blw_h) ''' 
				rsi_event_id = (-3, -2, 3, 0, 8, 7, -8)
				c_blw_l = rsi_event_id[0]
				blw_l = rsi_event_id[1]
				c_abv_l = rsi_event_id[2]
				in_band = rsi_event_id[3]
				c_abv_h = rsi_event_id[4]
				abv_h = rsi_event_id[5]
				c_blw_h = rsi_event_id[6]
				''' RSI_LowBand: cross_above = 3; cross_below = -3; below = -2; in_band = 0 '''
				rsi_lowBB_diff = diff(RSI, BB_l)
				rsi_lowBB_diff[rsi_lowBB_diff <= 0] = blw_l
				rsi_lowBB_diff[rsi_lowBB_diff > 0] = in_band
				rsi_lowBB_crossover = findCrossover(rsi_lowBB_diff, blw_l, c_abv_l, c_blw_l)
				''' RSI_HighBand: cross_above = 8; cross_below = -8; above = 7; in_band = 0'''
				rsi_highBB_diff = diff(RSI, BB_h)	
				rsi_highBB_diff[rsi_highBB_diff >= 0] = abv_h
				rsi_highBB_diff[rsi_highBB_diff < 0] = in_band
				rsi_highBB_crossover = findCrossover(rsi_highBB_diff, abv_h, c_abv_h, c_blw_h)
				''' RSI_EventList: combine all together '''
				rsi_eventArray = rsi_lowBB_crossover + rsi_highBB_crossover
				rsi_eventArray = np.append(np.array([0 for x in range(_rsi_len+_bb_len-1)]), rsi_eventArray)
				#########################################################################################
				if not any(rsi_eventArray) == 0:
					''' Setting the trading period '''
					valid_trading_period = np.zeros(len(df))
					if case == 'historical':
						valid_trading_period[_rsi_len + _bb_len:idx_end] = 1
					elif case == 'live':
						valid_trading_period[_rsi_len + _bb_len:idx_end+1] = 1	
					''' Setting Seasonal window '''
					entryDate = str(y_in) + convertFormat(MRCI.ix[i, 'i_Month'], MRCI.ix[i, 'i_Date'])
					exitDate = str(y_out) + convertFormat(MRCI.ix[i, 'o_Month'], MRCI.ix[i, 'o_Date'])
					season_start_date = stringDate_toDatetime(entryDate) - timedelta(days_before) 
					season_end_date = stringDate_toDatetime(exitDate) + timedelta(days_after)	
					seasonal_trading_period = np.zeros(len(df))	
					(idx_in, idx_in, idx_out, idx_out) = findDataRange(df, 'Date', season_start_date, season_end_date, 0, 0)
					#print "idx_in: ", idx_in
					#print "idx_out: ", idx_out 
					if idx_in < _rsi_len + _bb_len:
						seasonal_trading_period[(_rsi_len + _bb_len):idx_out] = 1
					else:
						seasonal_trading_period[idx_in:idx_out] = 1
					''' Initialize trading action list'''
					rsi_action = initStringList('none', len(df))
					''' Initialize position size and position PNL array '''
					posSize = np.zeros(len(df))
					posPrc = np.zeros(len(df))
					dynamicPNL = np.zeros(len(df))
					realizedPNL = np.zeros(len(df))
					''' Executing rsi strategy '''
					if follow_strat == 1:
						if strat == 'buy':
							for k in range(len(df)):
								#''' Long-only strategy '''
								strat_rsi_bb_crossover_1(k, df.Close, buy_symbol, _hard_stop ,avg_trueRange, valid_trading_period, \
									rsi_eventArray, rsi_event_id, rsi_action, posSize, posPrc, dynamicPNL, realizedPNL)
						elif strat == 'sell':	
							for k in range(len(df)):	
							#''' Short-only strategy '''
								strat_rsi_bb_crossover_2(k, df.Close, buy_symbol, _hard_stop ,avg_trueRange, valid_trading_period, \
									rsi_eventArray, rsi_event_id, rsi_action, posSize, posPrc, dynamicPNL, realizedPNL)		
					else:		
						for k in range(len(df)):
							#''' Long-Short strategy '''					
							strat_rsi_bb_crossover_3(k, df.Close, buy_symbol, _hard_stop ,avg_trueRange, valid_trading_period, \
								rsi_eventArray, rsi_event_id, rsi_action, posSize, posPrc, dynamicPNL, realizedPNL)
					''' Generate year summary ''' 
					yearSummary = DataFrame(index = np.arange(len(df)), columns=['TradeID', 'Year', 'Symbol', 'Strat', \
						'StartTrading', 'EndTrading', 'Date', 'DateTime', 'Close', 'IsPeriod', 'Action', \
									'PosSize', 'PosDir', 'PosPrc', 'DynamicPNL', 'RealizedPNL'])
					yearSummary.TradeID = tradeID
					yearSummary.Year = year
					yearSummary.Symbol = symbol
					yearSummary.Strat = strat
					#yearSummary.Buy = MRCI.ix[i, 'Buy'] + " " + MRCI.ix[i, 'b_Month'] + str(y_buy)
					#yearSummary.Sell = MRCI.ix[i, 'Sell'] + " " + MRCI.ix[i, 's_Month'] + str(y_sell)
					yearSummary.StartTrading = datetime_toString(start_trading_date)
					yearSummary.EndTrading = datetime_toString(end_trading_date)	
					yearSummary.Date = df.Date
					yearSummary.DateTime = dt
					#yearSummary.DateCode = t
					yearSummary.Close = df.Close
					yearSummary.IsPeriod = seasonal_trading_period
					yearSummary.Action = rsi_action
					yearSummary.PosSize = abs(posSize)
					yearSummary.PosDir = posSize
					yearSummary.PosPrc = posPrc
					yearSummary.DynamicPNL = dynamicPNL
					yearSummary.RealizedPNL = realizedPNL
					yearSummaryList.append(yearSummary)
	""" Error Check for a single trade"""
	if len(yearSummary) == []:
		print ("Check the year cases logic error for this trade")
		raise SystemExit 
	""" Generate a time-series trade summary for the trade """
	if yearSummaryList != []:
		tradeSummary = mergeResultList(yearSummaryList)
		tradeSummary = addDollarPNLColumn(buy_symbol, tradeSummary, 'DynamicPNL', 'DynamicDollarPNL')
		tradeSummary = addDollarPNLColumn(buy_symbol, tradeSummary, 'RealizedPNL', 'RealizedDollarPNL')
		tradeSummary = addDailyPNLChange(tradeSummary, 'DynamicDollarPNL', 'RealizedDollarPNL', 'DailyPNLChange')
		tradeSummary.DailyPNLChange[tradeSummary.Action == 'none'] = 0
		tradeSummaryList.append(tradeSummary)
		#cd "C:\Gary Yang\Dropbox\seasonal_Report\Test"
		#tradeSummary.to_csv('test.csv')
	################################################################################################

os.chdir("C:\\Users\\GYANG\\Google Drive\\Historical Data\\Sensonal_Fu_D")

#""" Generate a total Summary from all trades depends on # of Trades vs. PNL """
#totalSummary = mergeResultList(tradeSummaryList)
#totalSummary.to_csv('totalSummary.csv')
""" Generate time-series based data """
t_start = stringDate_toDatetime(str(backTestYearStart-1) + "0101")
t_end = datetime.now()
ts = TimeSeries(pd.date_range(t_start, t_end))
''' For plotting '''
t_code = getDateCode(ts) 
t_int = getDateInt(ts)
timeFrame_index = 'DateTime'
tradeSummaryList = set_timeFrameIndex(tradeSummaryList, timeFrame_index)

#''' get a certain tradeSummary to fit in timeFrame '''
#x = tradeEnd - tradeStart
#tf_tradeSummary = tradeSummary_fitTimeFrame(x, tradeSummaryList, ts, 'TradeID', 'Year', 'Symbol', 'StartTrading', 'EndTrading', 'Date', \
#	'Close', 'IsPeriod', 'Action', 'PosSize', 'PosDir', 'PosPrc', 'DynamicPNL', 'RealizedPNL', 'DynamicDollarPNL', 'RealizedDollarPNL', 'DailyPNLChange')	
#tf_tradeSummary.to_csv(str(x)+'.csv')

""" Generate Lists that fill the timeFrame """
	#startTradingList = []
	#endTradingList = []
TradeIDList = seriesData_fitTimeFrame(tradeSummaryList, 'TradeID', ts)
YearList = seriesData_fitTimeFrame(tradeSummaryList, 'Year', ts)
SymbolList = seriesData_fitTimeFrame(tradeSummaryList, 'Symbol', ts)
#BuyList = seriesData_fitTimeFrame(tradeSummaryList, 'Buy', ts)
#SellList = seriesData_fitTimeFrame(tradeSummaryList, 'Sell', ts)
StartTradingList = seriesData_fitTimeFrame(tradeSummaryList, 'StartTrading', ts)
EndTradingList = seriesData_fitTimeFrame(tradeSummaryList, 'EndTrading', ts)
#DateList = seriesData_fitTimeFrame(tradeSummaryList, 'Date', ts)
CloseList = seriesData_fitTimeFrame(tradeSummaryList, 'Close', ts)
IsPeriodList = seriesData_fitTimeFrame(tradeSummaryList, 'IsPeriod', ts)
PosSizeList = seriesData_fitTimeFrame(tradeSummaryList, 'PosSize', ts)
PosDirList = seriesData_fitTimeFrame(tradeSummaryList, 'PosDir', ts)
ActionList = seriesAction_fitTimeFrame(tradeSummaryList, 'Action', ts, PosDirList)
PosPrcList = seriesPosPrc_fitTimeFrame(tradeSummaryList, 'PosPrc', ts, PosDirList)
DynamicPNLList = seriesData_fitTimeFrame(tradeSummaryList, 'DynamicPNL', ts)
RealizedPNLList = seriesData_fitTimeFrame(tradeSummaryList, 'RealizedPNL', ts)
DynamicDollarPNLList = seriesData_fitTimeFrame(tradeSummaryList, 'DynamicDollarPNL', ts)
RealizedDollarPNLList = seriesData_fitTimeFrame(tradeSummaryList, 'RealizedDollarPNL', ts)
DailyPNLChangeList = seriesData_fitTimeFrame(tradeSummaryList, 'DailyPNLChange', ts)
#''' check the tradeSummary after fill to timeFrame '''
#tf_tradeSummary = tradeSummary_fillMissing(x, ts, TradeIDList, YearList, SymbolList, StartTradingList, EndTradingList, t_int, CloseList, \
#	IsPeriodList, ActionList, PosSizeList, PosDirList, PosPrcList, DynamicPNLList, RealizedPNLList, DynamicDollarPNLList, RealizedDollarPNLList, DailyPNLChangeList)
#tf_tradeSummary.to_csv(str(x)+'_1.csv')

tf_tradeSummaryList = []
for k in range(len(TradeIDList)):
	tf_tradeSummary = tradeSummary_fillMissing(k, ts, TradeIDList, YearList, SymbolList, StartTradingList, EndTradingList, t_int, CloseList, \
		IsPeriodList, ActionList, PosSizeList, PosDirList, PosPrcList, DynamicPNLList, RealizedPNLList, DynamicDollarPNLList, RealizedDollarPNLList, DailyPNLChangeList)
	tf_tradeSummaryList.append(tf_tradeSummary)

tf_totalSummary = mergeResultList(tf_tradeSummaryList)
tf_totalSummary = tf_totalSummary[tf_totalSummary.Action != 'none']
tf_totalSummary = tf_totalSummary[tf_totalSummary.TradeID != -99]
tf_totalSummary = tf_totalSummary.sort(['Date'])
tf_totalSummary.to_csv('DetailSummary.csv')

""" """
PosSizeAll = mergeDataList(PosSizeList)
IsPeriodAll = mergeDataList(IsPeriodList)
DynamicPNLAll = mergeDataList(DynamicDollarPNLList)

RealizedPNLAll = mergeDataList(RealizedDollarPNLList)

CumPNLAll = np.cumsum(RealizedPNLAll)

DailyPortfolioChange = mergeDataList(DailyPNLChangeList)
DailySummary = DataFrame(index = ts)
DailySummary['PosSizeAll'] = PosSizeAll
DailySummary['IsPeriodAll'] = IsPeriodAll
DailySummary['DynamicPNLAll'] = DynamicPNLAll
DailySummary['RealizedPNLAll'] = RealizedPNLAll
DailySummary['CombinePNL'] = DynamicPNLAll + RealizedPNLAll
DailySummary['DailyPortfolioChange'] = DailyPortfolioChange
DailySummary['CumPNL'] = CumPNLAll

DailySummary.to_csv('DailySummary.csv')


#CumPNLAll += DynamicPNLAll

#fig = figure(strat)
#ax1 = fig.add_subplot(2,1,1) 
#years = YearLocator()
#months = MonthLocator()
#yearFormatter = DateFormatter('%Y %m %d')  
#monthFormatter = DateFormatter('%m') 
#ax1.xaxis.set_major_locator(years)
#ax1.xaxis.set_minor_locator(months)
#ax1.xaxis.set_major_formatter(yearFormatter)
##ax1.xaxis.set_minor_formatter(monthFormatter)
#ax1.plot_date(t_code, CumPNLAll, marker = 'o', ms = 2, linestyle = '-', color = 'b')
#setp( gca().get_xticklabels(), rotation=45, horizontalalignment='right', fontsize = 10)
#
#ax2 = fig.add_subplot(2,1,2) 
#years = YearLocator()
#months = MonthLocator()
#yearFormatter = DateFormatter('%Y %m %d')  
#monthFormatter = DateFormatter('%m') 
#ax2.xaxis.set_major_locator(years)
#ax2.xaxis.set_minor_locator(months)
#ax2.xaxis.set_major_formatter(yearFormatter)
##ax2.xaxis.set_minor_formatter(monthFormatter)
#ax2.plot_date(t_code, PosSizeAll, marker = 'o', ms = 0, linestyle = '-', color = 'r')
#ax2.plot_date(t_code, IsPeriodAll, marker = 'o', ms = 0, linestyle = '-', color = 'b')
#setp(gca().get_xticklabels(), rotation=45, horizontalalignment='right', fontsize = 10)
#
#show()
#"""