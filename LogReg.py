import os
import math
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

def LogReg_ReadInputs(filepath):
    
    #function that reads all four of the Logistic Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    XTrainpath = filepath + "/LogReg_XTrain.csv"
    yTrainpath = filepath + "/LogReg_yTrain.csv"
    XTestpath = filepath + "/LogReg_XTest.csv"
    yTestpath = filepath + "/LogReg_yTest.csv"
    reader = csv.reader(open(XTrainpath, "rb"), delimiter=",")
    x = list(reader)
    XTrain = np.array(x).astype("int")
    reader1 = csv.reader(open(yTrainpath, "rb"), delimiter=",")
    y = list(reader1)
    yTrain = np.array(y).astype("int")
    reader2 = csv.reader(open(XTestpath, "rb"), delimiter=",")
    x1 = list(reader2)
    XTest = np.array(x1).astype("int")
    reader3 = csv.reader(open(yTestpath, "rb"), delimiter=",")
    y2 = list(reader3)
    yTest = np.array(y2).astype("int")
    '''
    print len(XTrain), len(XTrain[0])
    print len(yTrain), len(yTrain[0])
    print len(XTest), len(XTest[0])
    print len(yTest), len(yTest[0])
    '''
    feature = len(XTrain[0])
    XTrain1 = np.zeros(shape=(len(XTrain), feature + 1))
    XTest1 = np.zeros(shape=(len(XTest), feature + 1))
    # add 1 to each array first place
    for j in range(0, len(XTrain)):
        XTrain1[j] = np.concatenate(([1], XTrain[j]), axis=1)
    for j in range(0, len(XTest)):
        XTest1[j] = np.concatenate(([1], XTest[j]), axis=1)
    return (XTrain1, yTrain, XTest1, yTest)


def LogReg_CalcObj(X, y, w):
    
    #function that outputs the conditional log likelihood we want to maximize.

    #Input
    #w      : numpy weight vector of appropriate dimensions initialized to 0.5
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #cll   : The conditional log likelihood we want to maximize
    
    cll = 0
    yf = y.flatten()
    wf = w.flatten()
    N = len(X)
    for i in range(0, N):
        cll = cll + yf[i]*np.dot(wf, X[i]) - np.dot(wf, X[i]) - np.log(1 + np.exp(-np.dot(wf, X[i])))
    cll = cll/N
    return cll
    
def LogReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : weight vector 

    #Output
    #sg : gradient of the weight vector
    sg = np.zeros((len(w), 1))
    wf = w.flatten()
    xf = x.flatten()
    for i in range(0, len(w)):
        #print xf[i]
        #print np.dot(wf, xf)
        sg[i] = (y[0] - 1/(1 + np.exp(-np.dot(wf, xf)))) * xf[i]
    return sg
        
def LogReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.

    #Input
    #w  : weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient ascent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    w = w + eta * sg
    return w
    
def LogReg_PredictLabels(X, y, w):
    
    #Function that returns the value of the predicted y along with the number of
    #errors between your predictions and the true yTest values

    #Input
    #w : weight vector 
    #AND EITHER
    #XTest : nx(K+1) numpy matrix containing m number of d dimensional testing features
    #yTest : nx1 numpy vector containing the actual output for the testing features
    #OR
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    
    #Output
    #yPred : An nx1 vector of the predicted labels for yTest/yTrain
    #perMiscl : The percentage of y's misclassified
    
    yPred = []
    perMiscl = 0
    yf = y.flatten()
    wf = w.flatten()
    dif = 0.0
    for i in range(0, len(X)):
        p1 = 1/(1+np.exp(-np.dot(wf, X[i].flatten())))
        if p1 > 0.5:
            yPred.append(1)
        else:
            yPred.append(0)
        if yPred[0-1] != y[i]:
            dif += 1
    perMiscl = dif/len(X)
    #print perMiscl
    return (yPred, perMiscl)

def LogReg_SGA(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Ascent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #w             : final weight vector
    #trainPerMiscl : a vector of percentages of misclassifications on your training data at every 200 gradient descent iterations
    #testPerMiscl  : a vector of percentages of misclassifications on your testing data at every 200 gradient descent iterations
    #yPred         : a vector of your predictions for yTest using your final w
    
    trainPerMiscl = []
    testPerMiscl = []
    w = [0.5 for i in range(0, len(XTrain[0]))]
    w = np.array(w)
    w = np.reshape(w, (len(w), 1))
    N = 5
    iter = 1
    for j in range(0, N):
        for i in range(0, len(XTrain)):
            sg = LogReg_CalcSG(np.reshape(XTrain[i], (1, len(XTrain[i]))), yTrain[i], w)
            eta = 0.5 / (np.sqrt(iter))
            w = LogReg_UpdateParams(w, sg, eta)
            if iter%200 == 0:
                yeT = LogReg_PredictLabels(XTrain, yTrain, w)[1]
                yeP = LogReg_PredictLabels(XTest, yTest, w)[1]
                #print yeT, yeP
                trainPerMiscl.append(yeT)
                testPerMiscl.append(yeP)
            iter = iter + 1
    yPred = LogReg_PredictLabels(XTest, yTest, w)[0]
    #print yPred
    trainPerMiscl = np.reshape(np.array(trainPerMiscl), (len(trainPerMiscl), 1))
    testPerMiscl = np.reshape(np.array(testPerMiscl), (len(testPerMiscl), 1))
    #print trainPerMiscl, testPerMiscl
    return (w, trainPerMiscl, testPerMiscl, yPred)

    
def plot(trainPenMiscl, testPerMiscl):     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    t = [200*i for i in range(0, len(trainPerMiscl))]
    line1, = plt.plot(t, trainPenMiscl, 'b--', label='trainError')
    line2, = plt.plot(t, testPerMiscl, 'r--', label='testError')
    plt.legend(loc = 'upper right')
    plt.ylabel('Error')
    plt.xlabel('Epoch time')
    plt.show()
    return None

(XTrain, yTrain, XTest, yTest) = LogReg_ReadInputs("./data")
(w, trainPerMiscl, testPerMiscl, yPred) = LogReg_SGA(XTrain, yTrain, XTest, yTest)
plot(trainPerMiscl, testPerMiscl)