import numpy as np 
import random as rd
# 1. given R distribution : arrayPnl, arrayBins
# 2. compute SQN = (Expectancy / Standard Deviation of R) * sqrt (number of trades)
# 3. 

win1 = np.empty(70)
win1.fill(1)
win2 = np.empty(10)
win2.fill(2)
#win3 = np.empty(40)
#win3.fill(1)
loss1 = np.empty(10)
loss1.fill(-3.5)
loss2 = np.empty(10)
loss2.fill(-2)

pnl = win1
pnl = np.append(pnl, win2)
#pnl = np.append(pnl, win3)
pnl = np.append(pnl, loss1)
pnl = np.append(pnl, loss2)


def computeSQN(arrPnl):
	prob = 1/len(arrPnl)
	arrProb = np.empty(len(arrPnl))
	arrProb.fill(prob)

	expectancy = np.dot(arrPnl, arrProb)
	stdPnl = np.std(arrPnl)
	return (expectancy / stdPnl) * sqrt(len(arrPnl))








arr1 = np.array([1,2,3,4,5])
simCnt = 1000000
i = 0
r1 = r2 = r3 = r4 = r5 = 0

while (i < simCnt):
	result = rd.choice(arr1)
	if result == 1:
		r1 += 1
	elif result == 2:
		r2 += 1
	elif result == 3:
		r3 += 1
	elif result == 4:
		r4 += 1 
	else:
		r5 += 1
	i += 1

l = []
l.append(r1)
l.append(r2)
l.append(r3)
l.append(r4)
l.append(r5)
l




				
















	