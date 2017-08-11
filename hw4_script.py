import os
import csv
import numpy as np
import perceptron
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Point to data directory here
# By default, we are pointing to '../data/'

data_dir = os.path.join('.','data')

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

'''
# Visualize the image
idx = 0
datapoint = XTrain[idx, 1:]
plt.imshow(datapoint.reshape((28,28), order = 'F'), cmap='gray')
plt.show()
'''

(n, d) = XTrain.shape
# TODO: Test perceptron_predict function, defined in perceptron.py
#w = np.zeros((d,1))
#print perceptron.perceptron_predict(w, XTrain[0])

# TODO: Test perceptron_train function, defined in perceptron.py
w0 = [0 for i in range(0, len(XTrain[0]))]
# w0 = np.array(w0)
# # #print w0.shape
# w0 = perceptron.perceptron_train(w0, XTrain, yTrain, 10)
# err = 0
# for i in range(0, len(yTest)):
#     yhat = perceptron.perceptron_predict(w0, XTest[i])
#     if yhat != yTest[i]:
#         err += 1
# print float(err)/len(yTest)

#print w0

# TODO: Test RBF_kernel function, defined in perceptron.py
#sigma = 0.1
#print perceptron.RBF_kernel(XTrain[0], XTrain[1], sigma)

# TODO: Test kernel_perceptron_predict function, defined in perceptron.py
# x = XTrain[0]
# sigma = 0.1
# a = np.zeros((n,1))
#print perceptron.kernel_perceptron_predict(a, XTrain, yTrain, x, sigma)

# TODO: Test kernel_perceptron_train function, defined in perceptron.py
a0 = [0 for i in range(0, len(XTrain))]
sigma = 100
a0 = perceptron.kernel_perceptron_train(a0, XTrain, yTrain, 2, sigma)
err = 0
for i in range(0, len(yTest)):
    yhat = perceptron.kernel_perceptron_predict(a0, XTrain, yTrain, XTest[i], sigma)
    if yhat != yTest[i]:
        err += 1
print float(err)/len(yTest)

# TODO: Run experiments outlined in HW4 PDF
