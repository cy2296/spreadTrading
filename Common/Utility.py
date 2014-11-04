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


#def mergeFutSpreadFile(str_bSymbol, int_bMonth, str_sSymbol, int_sMonth):


