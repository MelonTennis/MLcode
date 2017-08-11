import os
import csv
import numpy as np
import NB
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test logProd function, defined in NB.py

# TODO: Test NB_XGivenY function, defined in NB.py

# TODO: Test NB_YPrior function, defined in NB.py

# TODO: Test NB_Classify function, defined in NB.py
alpha = 2
beta = 5
a = 5
b = 2
D = NB.NB_XGivenY(XTrain, yTrain, alpha, beta)
D1 = NB.NB_XGivenY(XTrain, yTrain, a, b)
Ds = NB.NB_XGivenY(XTrainSmall, yTrainSmall, alpha, beta)
print D
print D1
#
#B = integrate.quad(lambda x: np.power(x, alpha-1) * np.power(1-x, beta-1), 0, 1)
#f = lambda p: 1/B * np.power(p, alpha - 1)*np.power(1-p, beta - 1)
#Beta = [f(x) for x in D]
#plt.plot(Beta)
#plt.show()
#
p = NB.NB_YPrior(yTrain)
ps = NB.NB_YPrior(yTrainSmall)
res1 = NB.NB_Classify(D, p, XTest)
res2 = NB.NB_Classify(D1, p, XTest)
ress = NB.NB_Classify(Ds, ps, XTest)
print "error1", NB.classificationError(res1, yTest)
print "error2", NB.classificationError(res2, yTest)
print "small1", NB.classificationError(ress, yTest)
# TODO: Test classificationError function, defined in NB.py

# TODO: Run experiments outlined in HW2 PDF

