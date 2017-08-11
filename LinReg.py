import os
import math
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

def LinReg_ReadInputs(filepath):
    #function that reads all four of the Linear Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : NxK+1 numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nxK+1 numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    XTrainpath = filepath + "/LinReg_XTrain.csv"
    yTrainpath = filepath + "/LinReg_yTrain.csv"
    XTestpath = filepath + "/LinReg_XTest.csv"
    yTestpath = filepath + "/LinReg_yTest.csv"
    reader = csv.reader(open(XTrainpath, "rb"), delimiter=",")
    x = list(reader)
    XTrain = np.array(x).astype("float")
    reader1 = csv.reader(open(yTrainpath, "rb"), delimiter=",")
    y = list(reader1)
    yTrain = np.array(y).astype("float")
    reader2 = csv.reader(open(XTestpath, "rb"), delimiter=",")
    x1 = list(reader2)
    XTest = np.array(x1).astype("float")
    reader3 = csv.reader(open(yTestpath, "rb"), delimiter=",")
    y2 = list(reader3)
    yTest = np.array(y2).astype("float")
    '''
    print len(XTrain), len(XTrain[0])
    print len(yTrain), len(yTrain[0])
    print len(XTest), len(XTest[0])
    print len(yTest), len(yTest[0])
    '''
    # standardlize
    feature = len(XTrain[0])
    X = np.concatenate((XTrain, XTest), axis=0)
    maxX = [-sys.maxint for i in range(0, feature)]
    minX = [sys.maxint for i in range(0, feature)]
    for f in range(0, feature):
        for xx in range(0, len(X)):
            maxX[f] = max(X[xx][f], maxX[f])
            minX[f] = min(X[xx][f], minX[f])
    #print len(minX), len(maxX)

    #XTest.flags.writeable = True
    #print XTest
    for j in range(0, len(XTrain)):
        XTrain[j] = [(XTrain[j][i] - minX[i])/(maxX[i] - minX[i]) for i in range(0, feature)]
    for j in range(0, len(XTest)):
        XTest[j] = [(XTest[j][i] - minX[i])/(maxX[i] - minX[i]) for i in range(0, feature)]

    XTrain1 = np.zeros(shape=(len(XTrain), feature + 1))
    XTest1 = np.zeros(shape=(len(XTest), feature + 1))
    # add 1 to each array first place
    for j in range(0, len(XTrain)):
        XTrain1[j] = np.concatenate(([1], XTrain[j]), axis=1)
    for j in range(0, len(XTest)):
        XTest1[j] = np.concatenate(([1], XTest[j]), axis=1)
    return (XTrain1, yTrain, XTest1, yTest)

#LinReg_ReadInputs("./data")

def LinReg_CalcObj(X, y, w):
    
    #function that outputs the value of the loss function L(w) we want to minimize.

    #Input
    #w      : numpy weight vector of appropriate dimensions
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    
    #Output
    #loss   : The value of the loss function we want to minimize
    #print "CalcObj"
    lossVal = 0
    #Xf = X.flatten()
    yf = y.flatten()
    wf = w.flatten()
    for idx in range(0, len(X)):
        lossVal += math.pow(yf[idx] - np.dot(X[idx], wf), 2)
    lossVal = lossVal/len(X)
    #print len(y), len(X)
    #print "end CalcObj"
    return lossVal

def LinReg_CalcSG(x, y, w):

    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : (K+1)x1 dimensional weight vector 

    #Output
    #sg : gradient of the weight vector

    # sg(i) = -(yi - wTxi)xi
    #print "CalSG"
    sg = np.zeros((len(w), 1))
    wf = w.flatten()
    xf = x.flatten()
    for i in range(0, len(w)):
        sg[i][0] = (-2*(y[0] - np.dot(wf, xf))*xf[i])
    return sg

def LinReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.

    #Input
    #w  : (K+1)x1 dimensional weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient descent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    w = w - eta * sg
    return w
    
def LinReg_SGD(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Descent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional test features
    #yTest  : nx1 numpy vector containing the actual output for the test features
    
    #Output
    #w    : Updated Weight vector after completing the stochastic gradient descent
    #trainLoss : vector of training loss values at each epoch
    #testLoss : vector of test loss values at each epoch
    N = 100
    trainLoss = np.zeros((N, 1))
    testLoss = np.zeros((N, 1))
    w = [0.5 for i in range(0, len(XTrain[0]))]
    w = np.array(w)
    w = np.reshape(w, (len(w), 1))
    #w = np.array(wlist)
    iter = 1
    for i in range(0, N):
        trainLoss[i] = LinReg_CalcObj(XTrain, yTrain, w)
        testLoss[i] = LinReg_CalcObj(XTest, yTest, w)
        for r in range(0, len(XTrain)):
            eta = 0.5 / (np.sqrt(iter))
            xr = XTrain[r]
            yr = yTrain[r]
            sg = LinReg_CalcSG(np.reshape(xr, (1, len(xr))), yr,  w)
            w = LinReg_UpdateParams(w, sg, eta)
            iter = iter + 1
    return (w, trainLoss, testLoss)


def plot(trainLoss, testLoss):     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    #t = np.arrange(0, len(testLoss))
    line1, = plt.plot(trainLoss, 'b--', label='trainLoss')
    line2, = plt.plot(testLoss, 'r--', label='testLoss')
    plt.legend(loc = 'upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch time')
    plt.show()
    return None


(XTrain, yTrain, XTest, yTest) = LinReg_ReadInputs("./data")
(w, trainLoss, testLoss) = LinReg_SGD(XTrain, yTrain, XTest, yTest)
plot(trainLoss, testLoss)
