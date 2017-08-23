#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pylab

dataFile1 = 'Q2data.txt'
xPlot = [0 for j in range (200)]


def loadDataSet():
	points = []
	xVal = [[0 for i in range (2)] for j in range (200)]
	yVal = [[0 for i in range (1)] for j in range (200)]
	
	file = open(dataFile1, "r")
	index = 0
	for line in file:
		points.append(line.strip().split('	'))
		#print points
		xVal[index][0] = float(points[index][0])
		xVal[index][1] = float(points[index][1])
		xPlot[index] = float(points[index][1])
		yVal[index][0] = float(points[index][2])
		index = index+1
	file.close()
	

	plt.plot(xPlot,yVal, 'ro')
	#plt.show()
	return xVal, yVal

def standRegres(xVal, yVal):
	X = np.matrix(xVal)
	Y = np.matrix(yVal)
	theta = np.linalg.inv(X.T * X) * X.T * Y

	xx = np.linspace(0, 1)
	yy = np.array(theta[0] + theta[1]* xx)
	#plt.plot(xx, yy.T, color='b')
	#plt.show()
	#print theta
	return theta

def polyRegres(xVal, yVal):
	polyPoints = np.matrix(xVal)

	#add column into X
	#print polyPoints
	#print "Now adding x^2 column"
	newCol = np.power(xPlot, 2)
	newCol1 = np.power(xPlot, 3)
	newCol2 = np.power(xPlot, 4)
	newCol3 = np.power(xPlot, 5)
	newCol4 = np.power(xPlot, 6)
	newCol5 = np.power(xPlot, 7)
	newCol6 = np.power(xPlot, 8)
	newCol7 = np.power(xPlot, 9)

	polyPoints = np.column_stack((polyPoints, newCol, newCol1, newCol2, newCol3, newCol4, newCol5, newCol6, newCol7))
	#print polyPoints
	X = np.matrix(polyPoints)
	
	Y = np.matrix(yVal)

	thetaPrime = np.linalg.inv(X.T * X) * X.T * Y

	#just for testing
	#thetaPrime = np.polyfit(xVal, yVal, 4)

	#Plot the poly reg
	xx = np.linspace(0, 1)

	yy = np.array( thetaPrime[0] + 
	(thetaPrime[1]* xx) + 
	(thetaPrime[2] * (xx ** 2)) + 
	(thetaPrime[3] * (xx ** 3)) + 
	(thetaPrime[4] * (xx ** 4)) + 
	(thetaPrime[5] * (xx ** 5)) + 
	(thetaPrime[6] * (xx ** 6)) + 
	(thetaPrime[7] * (xx ** 7)) +
	(thetaPrime[8] * (xx ** 8)) +
	(thetaPrime[9] * (xx ** 9)) )
	plt.plot(xx, yy.T, color='g')
	print thetaPrime
	plt.show()
	return thetaPrime

xVal,yVal = loadDataSet()
theta = standRegres(xVal, yVal)
polyRegres(xVal, yVal)

