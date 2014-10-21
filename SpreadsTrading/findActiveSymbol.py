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

def findYearCase(symbol, m_buy, m_sell, m_in, m_out, year):
	if symbol == 'CL' or symbol == 'NG' or symbol == 'XRB' or symbol == 'HG':
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

def findTransferDate(symbol, y_buy, m_buy, y_sell, m_sell, days):
	if y_buy < y_sell: 
		y = y_buy
		m = m_buy
	elif y_buy > y_sell: 
		y = y_sell
		m = m_sell 
	else:
		y = y_buy = y_sell
		if m_buy < m_sell: m = m_buy
		else: m = m_sell
	if symbol == 'ZL' or symbol == 'ZS' or symbol == 'ZM' or symbol == 'ZC' or symbol == 'ZW' or symbol == 'KW' or symbol == 'LH': 	
		d = 15 - days
	elif symbol == 'GF'	or symbol == 'LE':
		d = 30 - days
	elif symbol == 'CL':
		d = 25 - days
		if m == 1: m = 12
		else: m -= 1
	elif symbol == 'HG' or symbol == 'HO' or symbol == 'NG' or symbol == 'XRB':
		d = 30 - days
		if m == 1: m = 12
		else: m -= 1
	yyyymmdd = str(201) + str(y) + convertFormat(m, d)	
	return yyyymmdd

def findEndTradingDate(symbol, year, month):
	if symbol == 'ZL' or symbol == 'ZS' or symbol == 'ZM' or symbol == 'ZC' or symbol == 'ZW' or symbol == 'GF' or symbol == 'LH': 
		date = 10
		end_trading_date = str(year) + convertFormat(month, date)
	elif symbol == 'CL' or symbol == 'HO' or symbol == 'NG' or symbol == 'XRB' or symbol =='LE':
		date =20
		if month == 1:
			end_trading_date = str(year) + convertFormat(12, date)
		else:
			end_trading_date = str(year) + convertFormat(month-1, date)
	elif symbol == 'HG':
		date = 5
		end_trading_date = str(year) + convertFormat(month, date)
	return stringDate_toDatetime(end_trading_date)



data = pd.read_csv("C:\\projects\\my\\Commodities\\MRCI.csv")
MRCI = DataFrame(data)

year = 2014
yearID = 4
''' define seasonal window '''
days_before = 30
days_after = 0
''' define early transfer '''
days_transfer = 7


yearBuy = []
yearSell = []
yearIn = []
yearOut = []
symbolList = []
fileList = []
caseList = []
startDateList = []
productList = []
endDateList = []
actionList = []

for i in range(len(MRCI)):
    buy_symbol = MRCI.ix[i, 'Buy']
    sell_symbol = MRCI.ix[i, 'Sell']
    if buy_symbol == sell_symbol:
        symbol = buy_symbol
    else: 
        print ("check symbol difference")
        raise SystemExit
    m_buy = month_toNumber(MRCI.ix[i, 'b_Month'])
    m_sell = month_toNumber(MRCI.ix[i, 's_Month'])
    m_in = MRCI.ix[i, 'i_Month']
    m_out = MRCI.ix[i, 'o_Month']
    m_out = MRCI.ix[i, 'o_Month']
    ### """ Generate symbol list that needs to be update """ ###	
    if MRCI.ix[i, 's_red'] == 1:
        y_in = y_out = y_buy = yearID
        y_sell = yearID + 1
    else:
        (y_buy, y_sell, y_in, y_out) = findYearCase(symbol, m_buy, m_sell, m_in, m_out, yearID)
    transferDate = findTransferDate(symbol, y_buy, m_buy, y_sell, m_sell, days_transfer)
    #print transferDate
    #end_trading_date = findEndTradingDate(buy_symbol, year, m_buy)
    #if datetime.now() <= end_trading_date:
    if datetime.now() <= stringDate_toDatetime(transferDate):
        ''' Before the transfer date '''
        eSignal_symbol = buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy) + ":" + sell_symbol + MRCI.ix[i, 's_Month'] + str(y_sell)  
        eSignal_file =  buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy) + "~" + sell_symbol + MRCI.ix[i, 's_Month'] + str(y_sell) + "_D.csv"
        case = 0
        season_startDate = stringDate_toDatetime(str(201) + str(y_in) + convertFormat(MRCI.ix[i, 'i_Month'], MRCI.ix[i, 'i_Date']))
        season_endDate = stringDate_toDatetime(str(201) + str(y_out) + convertFormat(MRCI.ix[i, 'o_Month'], MRCI.ix[i, 'o_Date']))
        season_startDate -= timedelta(days_before)
        season_endDate += timedelta(days_after)
    else:
        ''' After the transfer date '''
        eSignal_symbol = buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy+1) + ":" + sell_symbol + MRCI.ix[i, 's_Month'] + str(y_sell+1)
        eSignal_file = buy_symbol + " " + MRCI.ix[i, 'b_Month'] + str(y_buy+1) + "~" + sell_symbol + MRCI.ix[i, 's_Month'] + str(y_sell+1) + "_D.csv"
        case = 1
        season_startDate = stringDate_toDatetime(str(201) + str(y_in+1) + convertFormat(MRCI.ix[i, 'i_Month'], MRCI.ix[i, 'i_Date']))
        season_endDate = stringDate_toDatetime(str(201) + str(y_out+1) + convertFormat(MRCI.ix[i, 'o_Month'], MRCI.ix[i, 'o_Date']))
        season_startDate -= timedelta(days_before)
        season_endDate += timedelta(days_after)
    caseList.append(case)
    symbolList.append(eSignal_symbol)
    productList.append(buy_symbol)
    fileList.append(eSignal_file)
    startDateList.append(datetime_toString(season_startDate))
    endDateList.append(datetime_toString(season_endDate))
    actionList.append(MRCI.ix[i, 'action'])

testFrame = DataFrame(index = np.arange(len(caseList)), columns=['symbol', 'product', 'fileName', 'startDate', 'endDate', 'action'] )

testFrame.symbol = symbolList
testFrame.product = productList
testFrame.fileName = fileList
testFrame.startDate = startDateList
testFrame.endDate = endDateList
testFrame.action = actionList

#MRCI['eSignalSymbol'] = symbolList
#MRCI['fileNames'] = fileList
QCollectorList = DataFrame(index = np.arange(len(MRCI)))
QCollectorList['Symbol'] = symbolList

FileDataFrame = DataFrame(index = np.arange(len(MRCI)))
FileDataFrame['File'] = fileList

""" Output symbol list for QCollector download """

QCollectorList.to_csv("C:\\projects\\my\\Commodities\\active_symbol.csv")
#FileDataFrame.to_csv("C:\\projects\\my\\Commodities\\data\\process_files.csv")
testFrame.to_csv("C:\\projects\\my\\Commodities\\SpreadsTrading\\data\\process_files.csv")