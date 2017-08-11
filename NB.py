import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float
	log_product = float(sum(x))
	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters alpha and beta, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, alpha, beta):
	## Inputs ##
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	## Outputs ##
	# D - (2 by V) numpy ndarray
	n = XTrain.shape[0]
	V = XTrain.shape[1]
	D = np.zeros([2, XTrain.shape[1]])
	percent0 = [0 for x in range(0, V)]
	percent1 = [0 for x in range(0, V)]
	y1 = sum(yTrain)
	y0 = n - y1
	#B = integrate.quad(lambda x: np.power(x, alpha-1) * np.power(1-x, beta-1), 0, 1)
	#Beta = lambda p: 1/B * np.power(p, alpha - 1)*np.power(1-p, beta - 1)
	for i in range(0, n):
		label = yTrain[i]
		for j in range(0, V):
			if XTrain[i, j] == 1 and label == 1:
				percent1[j] = percent1[j] + 1
			elif XTrain[i, j] == 1 and label == 0:
				percent0[j] = percent0[j] + 1
	#print percent0, percent1
	for i in range(0, 2):
		for j in range(0, V):
			if i == 0:
				D[i, j] = float(percent0[j] + (alpha - 1))/(y0 + alpha + beta - 2)
			else:
				D[i, j] = float(percent1[j] + (alpha - 1))/(y1 + alpha + beta - 2)
			#print D[i, j]
	return D

# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float
	p = 0.0
	n = yTrain.shape[0]
	sum = np.sum(yTrain)
	p = float(n - sum)/n
	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m
	m = XTest.shape[0]
	V = XTest.shape[1]
	yHat = np.ones(XTest.shape[0])
	for i in range(0, m):
		temp0 = [0 for x in range(0, V)]
		temp1 = [0 for x in range(0, V)]
		for j in range(0, V):
			if XTest[i, j] == 0:
				temp0[j] = np.log(1-D[0, j])
				temp1[j] = np.log(1-D[1, j])
			else:
				temp0[j] = np.log(D[0,j])
				temp1[j] = np.log(D[1,j])
		p0 = np.log(p) + logProd(temp0)
		p1 = np.log(1-p) + logProd(temp1)
		if p0 < p1:
			yHat[i] = int(1)
		else:
			yHat[i] = int(0)
	return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float
	error = 0
	n = yHat.shape[0]
	for i in range(0, n):
		if yHat[i] != yTruth[i]:
			error += 1
	error = float(error)/n
	return error