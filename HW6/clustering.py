#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
#Your code here

def loadData(fileDj):
	data = []
	with open(fileDj) as f:
		tmpline = []
		for line in f.readlines():
			tmpline = line.split()
			data.append(tmpline[:len(tmpline)-1])
	data = np.array(data)
	#print data.shape
	#print data
	return data

## K-means functions 

def getInitialCentroids(X, k):
	#initialCentroids = []
	points = [[], []]
	keys = [x for x in range (k)]
	cents = []
	for i in range (0, k):
		newCent = []
		index = np.random.randint(0, len(X), size=1)
		newCent.append(float(X.item(index, 0)))
		newCent.append(float(X.item(index, 1)))
		cents.append(newCent)
	#print cents	
	cents = np.array(cents)
	#points = np.array(points)
	return cents, points

def getDistance(pt1,pt2):
	dist = 0
	x1 = float(pt1[0])
	y1 = float(pt1[1])
	x2 = float(pt2[0])
	y2 = float(pt2[1])
	#print x1, y1, x2, y2
	#Your code here
	dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
	return dist

def allocatePoints(X,clusters,points):
	#Your code here
	k = len(clusters)

	#Declare empty points so it doesn't aggregate
	points = [[], [], [], [], [], [], []]
	for i in range (0, len(X)):
		point = X[i]
		#print point[0]
		curDistance = 10000000
		assignedCluster = -1
		for j in range (0, k):
			curCluster = clusters[j]
			dist = getDistance(point, curCluster)
			#print "distance = ", dist," for k = ", j, "point = ", point, "cluster =", curCluster
			if (dist < curDistance):
				curDistance = dist
				assignedCluster = j
		#assign point to cluster
		#print assignedCluster
		newpoint = [point[0], point[1]]
		#print newpoint
		points[assignedCluster].append(newpoint)
		#np.append(points[assignedCluster], point)
				
	#print points
	#print len(points[0])
	#print len(points[1])
			
	return clusters,points

def updateCentroids(clusters,points):
	#Your code here
	#for each cluster
	for i in range (0, len(clusters)):
		size = len(points[i])
		#print "size = ", size
		#size = size of points for that cluster
		xavg = 0
		yavg = 0
		for j in range (0, size):
			#print "points[i][j] = ", points[i][j][0]
			xavg = xavg + float(points[i][j][0])
			yavg = yavg + float(points[i][j][1])
		xavg = xavg / size
		yavg = yavg / size
		newPoint = [xavg, yavg]
		clusters[i] = newPoint
	#print clusters
	return clusters,points


def visualizeClusters(clusters, points):
	#code here assumes k=2
	#convert to x and y arrays
	
	x1 = []
	x2 = []
	for i in range (0, len(points[0])):
		x1.append(points[0][i][0])
		x2.append(points[0][i][1])
	plt.scatter(x1, x2, color='red', marker='o', s=10)
	x1 = []
	x2 = []
	for i in range (0, len(points[1])):
		x1.append(points[1][i][0])
		x2.append(points[1][i][1])
	plt.scatter(x1, x2, color='blue', marker='o', s=10)
	x1 = []
	x2 = []
	for i in range (0, len(clusters)):
		x1.append(clusters[i][0])
		x2.append(clusters[i][1])
	#Cluster centroids
	plt.scatter(x1, x2, color='green', marker='^', s=40)
	plt.show()
	
	return clusters

def kmeans(X, k, maxIter=1000):
	#print "Calling kmeans(k=", k
	clusters, points = getInitialCentroids(X,k)
	#print "Starting clusters"
	#print clusters
	
	prevLen = 0
	for i in range (0, maxIter):
		clusters,points = allocatePoints(X,clusters,points)
		clusters,points = updateCentroids(clusters,points)
		#checks to see if membership numbers changed
		if(prevLen!=len(points[0])):
			prevLen = len(points[0])
		else:
			#print "ending on iteration: ", i
			break
		#print "iteration = ", i
	"""
	print "ending clusers ="
	print clusters
	print "size of each cluster"
	print len(points[0])
	print len(points[1])
	print len(points[2])
	print len(points[3])
	print len(points[4])
	print len(points[5])
	"""
	if(k==2):
		visualizeClusters(clusters, points)
	return clusters, points


def kneeFinding(X,kList):
	k_scores = [0, 0, 0, 0, 0, 0]
    	for idx, k in enumerate(kList):
		#print "//////////////// ITERATION = ", k," //////////////////////////"
		clusters, points = kmeans(X, k, 1000)
		#print points
		score = objF(clusters, points)
		#print "Obj score = ", score," for k = ", k
		k_scores[k-1] = score
	#plot k_scores
	plt.plot(kList, k_scores)
	plt.show()
	return
def objF(clusters, points):
	total_score = 0
	for j in range (0, len(clusters)):
		curCluster = clusters[j]
		#print curCluster
		ctotal = 0;
		for i in range (0, len(points[j])):
			xPoint = float(points[j][i][0])
			yPoint = float(points[j][i][1])
			#print "xPoint = ", type(xPoint)
			#print "yPoint = ", yPoint
			#print "curCluster = ", type(curCluster[0])
			ctotal = ctotal + (((xPoint - curCluster[0]) + (yPoint - curCluster[1]))**2)
		total_score = total_score + ctotal
	return total_score

def getPurity(X, clusters,points,fileDj):
	print "calculating purities"
	trueLabels = getLabels(fileDj)
	labels = [0] * len(X)
	for i in range (0, len(X)):
		curPoint = X[i]
		
		for j in range (0, len(points[0])):
			#print "curPoint->", curPoint
			#print "points[0][j]->", points[0][j]
			if(curPoint[0] == points[0][j][0] and curPoint[1] == points[0][j][1]):
				labels[i] = 1
		for j in range (0, len(points[1])):
			if(curPoint[0] == points[1][j][0] and curPoint[1] == points[1][j][1]):
				labels[i] = 2
	#print "labels = ", labels
	purities = []
	purities = purity(labels, trueLabels)
	print "Purities for k=2"
	print purities
	#Your code here
	return purities

def purity(labels, trueLabels):
	purities = [0, 0]
	tmp1 = 0
	tmp2 = 0
	n1 = 0
	n2 = 0
	#print type(labels[0])
	#print type(trueLabels[0][0])
	for i in range (0, len(labels)):
		if(trueLabels[i]==1):
			if(labels[i]==2):
				tmp1 = tmp1+1
			n1 = n1 + 1
		if(trueLabels[i]==2):
			if(labels[i]==1):
				tmp2 = tmp2+1
			n2 = n2+1
	purities[0] = float(tmp1) / float(n1)
	purities[1] = float(tmp2) / float(n2)
	return purities

def getLabels(fileDj):
	data = []
	with open(fileDj) as f:
		tmpline = []
		for line in f.readlines():
			tmpline = line.split()
			data.append(int(tmpline[len(tmpline)-1]))
	return data
## GMM functions 

#calculate the initial covariance matrix
#covType: diag, full
def getInitialsGMM(X,k,covType):
	if covType == 'full':
		dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
		covMat = np.cov(dataArray)
	else:
		covMatList = []
	for i in range(len(X[0])-1):
	    data = [pt[i] for pt in X]
	    cov = np.asscalar(np.cov(data))
	    covMatList.append(cov)
	covMat = np.diag(covMatList)

	initialClusters = {}
	#Your code here
	return initialClusters


def calcLogLikelihood(X,clusters,k):
	loglikelihood = 0
	#Your code here
	return loglikelihood

#E-step
def updateEStep(X,clusters,k):
	EMatrix = []
	#Your code here
	return EMatrix

#M-step
def updateMStep(X,clusters,EMatrix):
	#Your code here
	return clusters

def visualizeClustersGMM(X,labels,clusters,covType):
	#your code
	return

def gmmCluster(X, k, covType, maxIter=1000):
	#initial clusters
	clustersGMM = getInitialsGMM(X,k,covType)
	labels = []
	#Your code here
	visualizeClustersGMM(X,labels,clustersGMM,covType)
	return labels,clustersGMM


def purityGMM(X, clusters, labels):
	purities = []
	#Your code here
	return purities




def main():
	#######dataset path
	datadir = sys.argv[1]
	pathDataset1 = datadir+'/humanData.txt'
	pathDataset2 = datadir+'/audioData.txt'
	dataset1 = loadData(pathDataset1)
	dataset2 = loadData(pathDataset2)

	#Q3
	clusters, points = kmeans(dataset1, 2, maxIter=1000)
	getPurity(dataset1,clusters,points, pathDataset1)
	
	#Q4
	kneeFinding(dataset1,range(1,7))
	
	
	"""
	#Q7
	labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
	labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full')

	#Q8
	labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
	labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full')

	#Q9
	purities11 = purityGMM(dataset1, clustersGMM11, labels11)
	purities12 = purityGMM(dataset1, clustersGMM12, labels12)
	purities21 = purityGMM(dataset2, clustersGMM21, labels21)
	purities22 = purityGMM(dataset2, clustersGMM22, labels22)
	"""

if __name__ == "__main__":
	main()
