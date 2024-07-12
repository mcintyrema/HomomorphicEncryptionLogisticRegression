import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as pt
import statistics as stat
import math
import preprocessData as prep

def sigmoid(z):
    # calc exponential
    temp = (1+np.exp(-z))
    g = 1/temp
    return g


def costFunction(theta, X_train, y_train):
    m = len(X_train) # number of training samples
    # calculate z
    z = np.dot(X_train, theta)
    z = X_train @ theta
    # calculate the hypothesis H0(x), predicted probability y = 1 (diabetic)
    h = sigmoid(z)

    #compute summation
    y_train = y_train.reshape(-1, 1)
    #prevent error from taking log(0)
    epsilon = 1e-10
    #remove almost zero values from h
    h = np.clip(h, epsilon, 1 - epsilon)
    sum1 = -y_train*np.log(h) - (1-y_train)*np.log(1-h)

    product1 = (1/m)*sum1
    cost = np.sum(product1)
    return cost


def main():
    X = prep.preprocess_data()[0] #(13500, 10)
    y = prep.preprocess_data()[1]
    theta = np.zeros(X.shape[1]) #(10,)
    print(theta[0], X[0])
    print("Cost = ", costFunction(theta, X, y))

main()