#################################################################
# The algorithm seeks the active symbol list from the MRCI list #
#################################################################

import pandas as pd
from pandas import Series, DataFrame, TimeSeries
import numpy as np
from datetime import datetime, timedelta
import time
import os
import math
from sympy import Symbol, sqrt
from sympy.solvers import nsolve
import findActiveSymbol

def stringDate_toDatetime(string):
    return datetime.strptime(string, "%Y%m%d")    

def datetime_toString(dt):
    return dt.strftime("%Y%m%d")



def getDataFrame(fileName):
	if os.path.exists(fileName):
		data = pd.read_csv(fileName)
		df = DataFrame(data)
	else: 
		print (fileName + " doesn't exist")
		df = []
	return df

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
	changes = np.append(np.array([0]), changes)
	for i in range(len(changes)):
		if changes[i] == arr_val: # cross up
			arr[i] = 0
			changes[i] = c_up_val
		elif changes[i] == -arr_val:
			arr[i] = 0
			changes[i] = c_dwn_val
	return arr + changes

def getUpPrice(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb, bb_flag):
	# if close > close[-1]
	x = Symbol('x')
	dUp = x - closes[-1]
	dDn = 0
	dAvgUp = (dAvgUps[-1] * (len_rsi - 1) + dUp) / len_rsi
	dAvgDn = dAvgDns[-1] * (len_rsi - 1) / len_rsi 
	rsi = dAvgUp / (dAvgUp + dAvgDn) * 100
	mean = (np.sum(rsis[-len_bb+1:]) + rsi) / len_bb
	sum2 = 0
	for i in range(-len_bb+1, 0):
		sum2 += (rsis[i] - mean) ** 2 
	sum2 += (rsi - mean) ** 2
	std_dev = sqrt(sum2 / len_bb)
	if bb_flag == 'up':
		bb = mean + para_bb * std_dev
	elif bb_flag == 'low':	
		bb = mean - para_bb * std_dev
	return nsolve(rsi - bb, x, closes[-1])

def getDnPrice(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb, bb_flag):
	# if close < close[-1]
	x = Symbol('x')
	dUp = 0
	dDn = closes[-1] - x
	dAvgUp = dAvgUps[-1] * (len_rsi - 1) / len_rsi
	dAvgDn = (dAvgDns[-1] * (len_rsi - 1) + dDn)/ len_rsi 
	rsi = dAvgUp / (dAvgUp + dAvgDn) * 100
	mean = (np.sum(rsis[-len_bb+1:]) + rsi) / len_bb
	sum2 = 0
	for i in range(-len_bb+1, 0):
		sum2 += (rsis[i] - mean) ** 2 
	sum2 += (rsi - mean) ** 2
	std_dev = sqrt(sum2 / len_bb)
	if bb_flag == 'up':
		bb = mean + para_bb * std_dev
	elif bb_flag == 'low':	
		bb = mean - para_bb * std_dev
	return nsolve(rsi - bb, x, closes[-1])

def getPriceToBuy(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb):
	price = getUpPrice(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb, 'low') 
	if price < closes[-1]:
		price = getDnPrice(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb, 'low') 
	return price

def getPriceToSell(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb):
	price = getDnPrice(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb, 'up') 
	if price > closes[-1]:
		price = getUpPrice(closes, rsis, dAvgUps, dAvgDns, len_rsi, len_bb, para_bb, 'up') 
	return price

def getCloseTime(symbol):
	if symbol == 'ZL' or symbol == 'ZS' or symbol == 'ZM' or symbol == 'ZC' or symbol == 'ZW':
		closeTime = '14:15'
	elif symbol == 'GF' or symbol == 'LE' or symbol == 'LH':
		closeTime = '14:00'
	elif symbol == 'CL' or symbol == 'HO' or symbol == 'NG' or symbol == 'XRB':
		closeTime = '14:30'
	elif symbol == 'HG':
		closeTime = '13:00'
	return closeTime	

def isTrend(waitAct, action):
	if waitAct == 'wait for long' and action == 'buy': isTrend = 1
	elif waitAct == 'wait for long' and action == 'sell': isTrend = 0		
	elif waitAct == 'wait for short' and action == 'buy': isTrend = 0	
	elif waitAct == 'wait for short' and action == 'sell': isTrend = 1	
	return isTrend			

def isPeriod(dt_now, dt_start, dt_end):
	if dt_now >= dt_start and dt_now <= dt_end:
		return 1
	else:
		return 0
######################################################################################
""" Read the files download from QCollector """
######################################################################################

#cd "C:\Users\GYANG\Google Drive\Historical Data\Sensonal_Fu_D\Single_Months_D\daily_update"

''' setting parameters '''
_rsi_len = 10
_atr_len = 15
_bb_len = 15
_up_bb = 1.95
_dwn_bb = 1.95
""" Columns for Watch List DataFrame """
tradeID = []
eSignal_Symbol = []
Date = []
IsPeriod = []
RsiEvent = []
Action = []
IsTrend = []
Price = []
CloseTime = []
count = 0

#data = pd.read_csv("C:\\projects\\my\\Commodities\\data\\process_files.csv")
df_file = DataFrame(pd.read_csv("C:\\projects\\my\\Commodities\\data\\process_files.csv"))
#fileList = df_file['File']
#f = fileList[0]
#df = getDataFrame(f)

for i in range(len(df_file)):
    f = df_file.ix[i, 'fileName']
    #fileList[i]
    print (f)
    print (type(f))
    df = DataFrame(pd.read_csv("C:\\projects\\my\\Commodities\\data\\" + f))
    print (type(df))
    if len(df) > _rsi_len + _bb_len:
        if df.ix[len(df)-1, 'Date'] == int(datetime_toString(datetime.now())):
            print (df.ix[len(df)-1, 'Date'])
            print (int(datetime_toString(datetime.now())))
            print ("Hi")
            df = df[:len(df)-1]	# elimate the last line 
        print (df.ix[len(df)-1, 'Date'])
        ###################################################################################
        arr_closes = np.asarray(df.Close)
        (RSI, dAvgUps, dAvgDns) = rsi(arr_closes, _rsi_len)
        (BB_h, BB_m, BB_l) = bb(RSI, _bb_len, _up_bb, _dwn_bb)
        ###################################################################################
        """"""""""""""""""""""""""""""""" RSI EVENT ARRAY """""""""""""""""""""""""""""""""
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
        #rsi_event_dict = {'c_blw_l': -3, 'blw_l': -2, 'c_abv_l': 3, 'in_band': 0, 'c_abv_h': 8, 'abv_h': 7, 'c_blw_h': -8}
        ''' RSI_LowBand: cross_above = 3; cross_below = -3; below = -2; in_band = 0 '''
        rsi_lowBB_diff = diff(RSI, BB_l)
        rsi_lowBB_diff[rsi_lowBB_diff <= 0] = blw_l
        rsi_lowBB_diff[rsi_lowBB_diff > 0] = in_band
        rsi_lowBB_crossover = findCrossover(rsi_lowBB_diff, blw_l, c_abv_l, c_blw_l)
        ''' RSI_HighBand: cross_above = 8; cross_below = -8; above = 7; in_band = 0 '''
        rsi_highBB_diff = diff(RSI, BB_h)	
        rsi_highBB_diff[rsi_highBB_diff >= 0] = abv_h
        rsi_highBB_diff[rsi_highBB_diff < 0] = in_band
        rsi_highBB_crossover = findCrossover(rsi_highBB_diff, abv_h, c_abv_h, c_blw_h)
        ''' RSI_EventList: combine all together '''
        rsi_eventArray = rsi_lowBB_crossover + rsi_highBB_crossover
        ''' Adjust the lenght '''
        RSI = np.append(np.array([RSI[0] for x in range(_rsi_len)]), RSI)
        dAvgUps = np.append(np.array([dAvgUps[0] for x in range(_rsi_len)]), dAvgUps)
        dAvgDns = np.append(np.array([dAvgDns[0] for x in range(_rsi_len)]), dAvgDns)
        BB_h = np.append(np.array([BB_h[0] for x in range(_rsi_len + _bb_len-1)]), BB_h)
        BB_m = np.append(np.array([BB_m[0] for x in range(_rsi_len + _bb_len-1)]), BB_m)
        BB_l = np.append(np.array([BB_l[0] for x in range(_rsi_len + _bb_len-1)]), BB_l)
        rsi_eventArray = np.append(np.array([0 for x in range(_rsi_len + _bb_len-1)]), rsi_eventArray)
        #####################################################################################
        if rsi_eventArray[-1] == c_blw_l or rsi_eventArray[-1] == c_abv_h or rsi_eventArray[-1] == blw_l or rsi_eventArray[-1] == abv_h:
            #print i
            count += 1
            tradeID.append(i)
            eSignal_Symbol.append(df_file.ix[i, 'symbol'])

  
            CloseTime.append(getCloseTime(df_file.ix[i, 'product']))

            Date.append(df.ix[len(df)-1, 'Date'])

            IsPeriod.append(isPeriod(datetime.now(), stringDate_toDatetime(str(df_file.ix[i, 'startDate'])), stringDate_toDatetime(str(df_file.ix[i, 'endDate']))))

            #print rsi_eventArray[-1]
            if rsi_eventArray[-1] == c_blw_l:
                act = 'wait for long'
                IsTrend.append(isTrend(act, df_file.ix[i, 'action']))
                RsiEvent.append('cross below low') 
                Action.append(act)
                Price.append(getPriceToBuy(arr_closes, RSI, dAvgUps, dAvgDns, _rsi_len, _bb_len, _dwn_bb))
            elif rsi_eventArray[-1] == blw_l:
                act = 'wait for long'
                IsTrend.append(isTrend(act, df_file.ix[i, 'action']))
                RsiEvent.append('below low') 
                Action.append(act)			
                Price.append(getPriceToBuy(arr_closes, RSI, dAvgUps, dAvgDns, _rsi_len, _bb_len, _dwn_bb))	
            elif rsi_eventArray[-1] == c_abv_h:
                act = 'wait for short'
                IsTrend.append(isTrend(act, df_file.ix[i, 'action']))
                RsiEvent.append('cross above high') 
                Action.append(act)
                Price.append(getPriceToSell(arr_closes, RSI, dAvgUps, dAvgDns, _rsi_len, _bb_len, _up_bb))	
            elif rsi_eventArray[-1] == abv_h:
                act = 'wait for short'
                IsTrend.append(isTrend(act, df_file.ix[i, 'action']))
                RsiEvent.append('cross below high') 
                Action.append(act)		
                Price.append(getPriceToSell(arr_closes, RSI, dAvgUps, dAvgDns, _rsi_len, _bb_len, _up_bb))	

    WatchList = DataFrame(index = np.arange(count))
    WatchList['tradeID'] = tradeID
    WatchList['eSignal_Symbol'] = eSignal_Symbol
    WatchList['Date'] = Date
    WatchList['IsPeriod'] = IsPeriod
    WatchList['IsTrend'] = IsTrend
    WatchList['RsiEvent'] = RsiEvent
    WatchList['Action'] = Action
    WatchList['Price'] = Price
    WatchList['CloseTime'] = CloseTime

    #cd "C:\\projects\\my\\Commodities\\Real_Time_Report"

    WatchList.to_csv("C:\\projects\\my\\Commodities\\WatchList.csv")
