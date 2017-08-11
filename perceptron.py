import numpy as np
import math

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################


def perceptron_predict(w, x):
  #print "1"
  # Input:
  #   w is the weight vector (d,),  1-d array
  #   x is feature values for test example (d,), 1-d array
  # Output:
  #   the predicted label for x, scalar -1 or 1
  wf = w.flatten()
  xf = x.flatten()
  yhat = np.dot(wf, xf)
  if yhat > 0:
    return 1
  else:
    return -1


def perceptron_train(w0, XTrain, yTrain, num_epoch):
  #print "2"
  # Input:
  #   w0 is the initial weight vector (d,), 1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   num_epoch is the number of times to go through the data, scalar
  # Output:
  #   the trained weight vector, (d,), 1-d array
  #print(w0)
  for iter in range(0, num_epoch):
    for i in range(0, len(XTrain)):
        yhat = perceptron_predict(w0, XTrain[i])
        if yhat != yTrain[i]:
          xr = np.reshape(XTrain[i], w0.shape)
          w0 = w0 + yTrain[i]*XTrain[i]
  return w0


def RBF_kernel(X1, X2, sigma):
  # Input:
  #   X1 is a feature matrix (n,d), 2-d array or 1-d array (d,) when n = 1
  #   X2 is a feature matrix (m,d), 2-d array or 1-d array (d,) when m = 1
    #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   K is a kernel matrix (n,m), 2-d array

  #----------------------------------------------------------------------------------------------
  # Special notes: numpy will automatically convert one column/row of a 2d array to 1d array
  #                which is  unexpected in the implementation
  #                make sure you always return a 2-d array even n = 1 or m = 1
  #                your implementation should work when X1, X2 are either 2d array or 1d array
  #                we provide you with some starter codes to make your life easier
  #---------------------------------------------------------------------------------------------
  if len(X1.shape) == 2:
    n = X1.shape[0]
  else:
    n = 1
    X1 = np.reshape(X1, (1, X1.shape[0]))
  if len(X2.shape) == 2:
    m = X2.shape[0]
  else:
    m = 1  
    X2 = np.reshape(X2, (1, X2.shape[0]))
  K = np.zeros((n,m))
  for i in range(0, n):
    for j in range(0, m):
      K[i][j] = np.exp(-np.dot(X1[i] - X2[j], X1[i] - X2[j])/(2*sigma*sigma))
  return K

def kernel_perceptron_predict(a, XTrain, yTrain, x, sigma):
  # Input:
  #   a is the counting vector (n,),  1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   x is feature values for test example (d,), 1-d array
  #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   the predicted label for x, scalar -1 or 1
  K = RBF_kernel(XTrain, x, sigma)
  ylabel = np.array([a[i]*yTrain[i] for i in range(0, len(a))])
  xp = np.dot(ylabel.flatten(), K.flatten())
  if xp <= 0:
    return -1
  else:
    return 1
 
def kernel_perceptron_train(a0, XTrain, yTrain, num_epoch, sigma):
  # Input:
  #   a0 is the initial counting vector (n,), 1-d array
  #   XTrain is feature values of training examples (n,d), 2-d array
  #   yTrain is labels of training examples (n,), 1-d array
  #   num_epoch is the number of times to go through the data, scalar
  #   sigma is the parameter $\sigma$ in RBF function, scalar
  # Output:
  #   the trained counting vector, (n,), 1-d array
  for iter in range(0, num_epoch):
    for i in range(0, len(XTrain)):
      yhat = kernel_perceptron_predict(a0, XTrain, yTrain, XTrain[i], sigma)
      if yhat != yTrain[i]:
        a0[i] = a0[i] + 1
  return a0