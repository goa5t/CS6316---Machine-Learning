#!/usr/bin/env python

import sys
import os
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA

def test_error(ytrue, yhat):
	
	if(len(ytrue) != len(yhat)):
		print "unequal parameters"
		return
	totalwrong = 0
	totalnum = len(ytrue)
	for i in range (0, len(ytrue)):
		#print ytrue[i]
		#print int(float(yhat[i]))
		#print "-----------"
		if(abs(int(ytrue[i])-int(yhat[i])) > 0):
			totalwrong = totalwrong + 1
	#print totalwrong
	#print totalnum
	error = float(totalwrong) / float(totalnum)
	return error
		
def loadData(train, test):
	Xtrain = []
	xsample=[]
	ytrain = []
	tmpline = []
	Xtest = []
	ytest = []
	## Load training matrices
    	with open(train) as f:
		tmpline = [0] * 257
		for line in f.readlines():
			tmpline = line.split()
			ytrain.append(tmpline[0])
			xsample = (tmpline[1:])
			Xtrain.append(xsample)
			xsample = [0] * 256
	Xtrain = np.array(Xtrain)
	ytrain = np.array(ytrain)
	
	#Load testing matrices
	with open(test) as f:
		tmpline = [0] * 257
		for line in f.readlines():
			tmpline = line.split()
			ytest.append(tmpline[0])
			xsample = (tmpline[1:])
			Xtest.append(xsample)
			xsample = [0] * 256
	Xtest = np.array(Xtest)
	ytest = np.array(ytest)
	#convert from strings to floats
	Xtrain = Xtrain.astype(np.float)
	Xtest = Xtest.astype(np.float)
	ytrain = ytrain.astype(np.float)
	ytest = ytest.astype(np.float)
	return Xtrain, ytrain, Xtest, ytest
	
def decision_tree(train, test):
    	y = []
	Xtrain, ytrain, Xtest, ytest = loadData(train, test)

	#Make classifier
	clf = DecisionTreeClassifier(max_features=250, max_depth=None, min_samples_split=5, class_weight="balanced", random_state=0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for dtree:"
	print error
	
    	return y

def knn(train, test):
	    	y = []
	    	Xtrain, ytrain, Xtest, ytest = loadData(train, test)

		#Make classifier
		clf = KNeighborsClassifier(n_neighbors = 3)
		clf.fit(Xtrain, ytrain)
		y1 = clf.predict(Xtrain)
		y = clf.predict(Xtest)
		terror = test_error(ytrain, y1)
		print "training error for KNN, k=3:"
		print terror
		error = test_error(ytest, y)
		print "test error for KNN, k=3:"
		print error
		print "\\\\\\\\\\\\\\\\"

		clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
		clf.fit(Xtrain, ytrain)
		y1 = clf.predict(Xtrain)
		y = clf.predict(Xtest)
		terror = test_error(ytrain, y1)
		print "training error for KNN, k=3, distance:"
		print terror
		error = test_error(ytest, y)
		print "test error for KNN, k=3, distance:"
		print error
		print "\\\\\\\\\\\\\\\\"

		clf = KNeighborsClassifier(n_neighbors = 4)
		clf.fit(Xtrain, ytrain)
		y1 = clf.predict(Xtrain)
		y = clf.predict(Xtest)
		terror = test_error(ytrain, y1)
		print "training error for KNN, k=7:"
		print terror
		error = test_error(ytest, y)
		print "test error for KNN, k=7:"
		print error
		print "\\\\\\\\\\\\\\\\"

		clf = KNeighborsClassifier(n_neighbors = 5)
		clf.fit(Xtrain, ytrain)
		y1 = clf.predict(Xtrain)
		y = clf.predict(Xtest)
		terror = test_error(ytrain, y1)
		print "training error for KNN, k=5:"
		print terror
		error = test_error(ytest, y)
		print "test error for KNN, k=5:"
		print error
		print "\\\\\\\\\\\\\\\\"

		clf = KNeighborsClassifier(n_neighbors = 5, weights='distance')
		clf.fit(Xtrain, ytrain)
		y1 = clf.predict(Xtrain)
		y = clf.predict(Xtest)
		terror = test_error(ytrain, y1)
		print "training error for KNN, k=5, d:"
		print terror
		error = test_error(ytest, y)
		print "test error for KNN, k=5, d:"
		print error
		print "\\\\\\\\\\\\\\\\"
	   	return y

def svm(train, test):
	y = []
	Xtrain, ytrain, Xtest, ytest = loadData(train, test)

	#Make classifier
	clf = SVC(kernel = 'rbf', C=7.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = rbf, C=7:"
	print error
	print "\\\\\\\\\\\\\\\\"
	#Make classifier
	clf = SVC(kernel = 'rbf', C=5.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = rbf, C=5:"
	print error
	print "\\\\\\\\\\\\\\\\"
	#Make classifier
	clf = SVC(kernel = 'poly', degree = 5, C=5.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = poly, deg = 5, C=5.0:"
	print error
	print "\\\\\\\\\\\\\\\\"
	#Make classifier
	clf = SVC(kernel = 'poly', degree = 5, C=7.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = poly, deg = 5, C=7.0:"
	print error
	print "\\\\\\\\\\\\\\\\"
	#Make classifier
	clf = SVC(kernel = 'poly', degree = 4, C=7.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = poly, deg = 4, C=7.0:"
	print error
	print "\\\\\\\\\\\\\\\\"
	#Make classifier
	clf = SVC(kernel = 'poly', degree = 4, C=5.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = poly, deg = 4, C=5.0:"
	print error
	print "\\\\\\\\\\\\\\\\"
	
	return y

def pca_knn(train, test):
	y = []
	Xtrain, ytrain, Xtest, ytest = loadData(train, test)

	#PCA, fit and transform
	pca = RandomizedPCA(n_components = 200)
	pca.fit(Xtrain)
	Xtrain = pca.transform(Xtrain)
	new_Xtest = pca.transform(Xtest)

	#Make classifier
	clf = KNeighborsClassifier(n_neighbors = 3)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(new_Xtest)

	#y1 = clf.predict(Xtrain)
	#terror = test_error(ytrain, y1)
	#print "training error for KNN, k=3:"
	#print terror

	error = test_error(ytest, y)
	print "test error for KNN, k=3:"
	print error
	print "\\\\\\\\\\\\\\\\"
	
    	return y

def pca_svm(train, test):
	y = []
	Xtrain, ytrain, Xtest, ytest = loadData(train, test)
	#PCA, fit and transform
	pca = RandomizedPCA(n_components = 50)
	pca.fit(Xtrain)
	Xtrain = pca.transform(Xtrain)
	new_Xtest = pca.transform(Xtest)

	
	#Make classifier
	clf = SVC(kernel = 'poly', degree = 4, C=7.0)
	clf.fit(Xtrain, ytrain)
	y = clf.predict(new_Xtest)
	error = test_error(ytest, y)
	print "test error for svm, kernel = poly, deg = 4, C=7.0:"
	print error
	
    	return y

if __name__ == '__main__':
	    model = sys.argv[1]
	    train = sys.argv[2]
	    test = sys.argv[3]

	    if model == "dtree":
		print(decision_tree(train, test))
	    elif model == "knn":
		print(knn(train, test))
	    elif model == "svm":
		print(svm(train, test))
	    elif model == "pcaknn":
		print(pca_knn(train, test))
	    elif model == "pcasvm":
		print(pca_svm(train, test))
	    else:
		print("Invalid method selected!")
