##############################################################################
#### Given today's date, generate a symbol.txt for QCollector to download ####
##############################################################################
InstruList = ['CL', 'GF', 'HG', 'HO', 'ZW', 'LE', 'LH', 'NG', 'XRB', 'ZC', 
			  'ZL', 'ZM', 'ZS', 'ZO', 'ES', 'NQ', '6A', '6B', '6C', '6E', 
			  '6J', '6S', 'GC', 'SI', 'ZT', 'ZF', 'ZN', 'ZB'] 

SymbolList = []

begYear = 1998
endYear = 2015
currYear = 2014
currMonth = 10

for i in range(len(InstruList)):
	for j in range(begYear, endYear):
		if j == currYear:
			print (i, j)



