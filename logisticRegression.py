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
    theta = np.array(theta)
    #make sure one column vector
    theta = theta.reshape(-1, 1)

    m = len(X_train)
    # calculate z
    z = np.dot(X_train, theta)
    h = sigmoid(z)

    #compute summation
    y_train = y_train.reshape(-1, 1)
    #prevent error from taking log(0)
    nozero = 1e-10
    #remove almost zero values from h
    h = np.clip(h, nozero, 1 - nozero)
    sum1 = -y_train*np.log(h) - (1-y_train)*np.log(1-h)

    product1 = (1/m)*sum1
    cost = np.sum(product1)
    return cost


def main():
    X = prep.preprocess_data()[0]
    y = prep.preprocess_data()[1]
    theta = [[2],
             [0],  
             [0]]
    #TODO: Get theta values of correct dimension
    print("Cost = ", costFunction(theta, X, y))

main()