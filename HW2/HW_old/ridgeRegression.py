#!/usr/bin/env python

import numpy as np
from sklearn import datasets, linear_model, cross_validation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
import pylab

dataFile1 = 'RRdata.txt'
points = []


def loadDataSet():
	xVal = [[0 for i in range (3)] for j in range (200)]
	yVal = [[0 for i in range (1)] for j in range (200)]
	file = open(dataFile1, "r")
	index = 0
	for line in file:
		points.append(line.strip().split(' '))
		#print points
		xVal[index][0] = float(points[index][0])
		xVal[index][1] = float(points[index][1])
		xVal[index][2] = float(points[index][2])
		yVal[index][0] = float(points[index][3])
		index = index+1
	file.close()
	size = index
	"""
	print "length is ->"
	print size
	print xVal
	print yVal
	"""
	return xVal, yVal

def ridgeRegress(xVal, yVal, lmbda, best):
	X = np.matrix(xVal)
	Y = np.matrix(yVal)
	
	#using the Identity matrix with 0 for B0, to avoid regularizing B0
	betaLR =  np.linalg.inv(X.T * X + (lmbda * np.matrix('0 0 0; 0 1 0; 0 0 1') ) )  * X.T * Y
	#betaLR = np.matrix(' 3; 1; 1')
	
	#for testing...
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(X, Y)

	# The coefficients
	#print('Coefficients: \n', regr.coef_)
	if best > 0:
		print betaLR
		#scatter
		xVal = np.matrix(xVal)
		xVal = np.delete(xVal, 0, 1)
		x1 = xVal[:,0]
		x2 = xVal[:,1]
		#print x1
		#print x2
		ax.scatter(x1, x2, yVal, color='r')
	
		xx = np.arange(-8, 8, .5)
		yy = np.arange(-8, 8, .5)
		Z = betaLR.item(0) + betaLR.item(1) * xx + betaLR.item(2) * yy
	#print Z
	
		xx, yy = np.meshgrid(xx, yy)
		ax.plot_surface(xx,yy,Z)
		plt.show()
	return betaLR

def cv(xVal, yVal):
	allLmbda = np.arange(0.02, 1.02, 0.02)
	#print allLmbda
	#set random seed
	np.random.seed(37)
	randomize = np.arange(0, 200, 1)
	#print randomize
	np.random.shuffle(randomize)
	"""
	print randomize
	print len(randomize)
	print xVal
	print yVal 
	print len(xVal)
	print len(yVal)
	print "randomizing.."
	"""
	for i in range(200):
		xVal[i] = xVal[randomize[i]]
		yVal[i] = yVal[randomize[i]]
	#print xVal
	#print yVal 
	
	#loop through lambda values
	lCounter = 0;
	MSEperL = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for lmbda in allLmbda:
		totalMSE = [0,0,0,0,0,0,0,0,0,0]
		#print lmbda
		#for each fold
		for i in range (10):
			#print "On lambda= ",lmbda," fold= ",i
			xTrain = []	
			yTrain = []
			xTest = xVal[i*20:20*i+20]
			yTest = yVal[i*20:20*i+20]
			if i < 1:
				xTrain = xVal[20:200]
				yTrain = yVal[20:200]
			elif i > 8:
				xTrain = xVal[0:180]
				yTrain = yVal[0:180]
			else:
				xTrain = xVal[0:i*20] + xVal[i*20+20:200]
				yTrain = yVal[0:i*20] + yVal[i*20+20:200]
			"""
			print xTest
			print len(xTest)
			print yTest
			print len(yTest)
			print "training sets"
			print len(xTrain)
			print len(yTrain)
			"""

			betaK = ridgeRegress(xTrain, yTrain, lmbda, 0)
			#print "JFLKJSLDKFSDF betaK = ",betaK
			#calculate MSE for Bk
			tmpSum = 0
			for j in range (20):
				yHat = betaK.item(0) + betaK.item(1) *xTest[j][1] + betaK.item(2) * xTest[j][2]
				
				#print "yHat is ",yHat
				#print yTest[j][0]
				difference = yTest[j][0] - yHat
				difference = difference * difference
				tmpSum = tmpSum + difference
				#print difference
			totalMSE[i] = tmpSum / 20
			#print "total MSE for lambda=",lmbda," fold=",i," is ",totalMSE[i]
		tmp = 0
		for i in range (10):
			tmp = tmp + totalMSE[i]
			#set the average MSE for this lambda to the array
			MSEperL[lCounter] = tmp / 10
		#print "MSE for lambda: ",lmbda," is ",MSEperL[lCounter]
		lCounter=lCounter+1
		
	print MSEperL
	minIndex = np.argmin(MSEperL)
	print "MIN: Lambda = ",allLmbda[minIndex]," MSE = ",MSEperL[minIndex]
	#plt.plot(allLmbda, MSEperL)
	#plt.show()
	#return the lambda with the lowest average MSE
	return allLmbda[minIndex]
			
xVal, yVal = loadDataSet()
ridgeRegress(xVal, yVal, 0, 1)
#lmbdaBest = cv(xVal, yVal)
#print lmbdaBest
#betaRR = ridgeRegress(xVal, yVal, lmbdaBest, 1)
#print betaRR

