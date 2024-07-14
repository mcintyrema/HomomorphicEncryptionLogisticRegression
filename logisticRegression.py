import numpy as np
import matplotlib.pyplot as plt
import preprocessData as prep


def sigmoid(z):
    # calc exponential
    temp = (1+np.exp(-z))
    g = 1/temp
    return g


def costFunction(theta, b, X_train, y_train):
    m = len(X_train) # number of training samples

    ####forward propagation###
    # calculate z
    z = np.dot(X_train, theta) + b
    # calculate the hypothesis H0(x), predicted probability y = 1 (diabetic)
    h = sigmoid(z)  #(13500,)
    #stabilize data for logarithms
    y_train = y_train.reshape(-1, 1) #(13500,)
    epsilon = 1e-10
    #remove almost zero and 1 values from h
    h = np.clip(h, epsilon, 1 - epsilon)
    print(y_train.shape, h.shape)
    sum0 = -y_train*np.log(h)
    sum1 = sum0 - (1-y_train)*np.log(1-h)
    # product1 = (1/m)*sum1
    cost = (np.sum(sum1)) * (1/m)

    ###back propagation###
    # bias = np.sum(h - y_train.flatten()) / m
    bias = np.sum(h - y_train) / m
    gradient = (1/m)*np.dot(X_train.T, (h-y_train))

    return cost, np.array(gradient), bias


def grad_descent(X_train, y_train, alpha, iterations):
    m = X_train.shape[0]  # Number of training examples
    
    # Initialize weights and bias
    theta, b = initialize_weights_and_bias(X_train.shape[1])
    costs = []
    cost_list2 = []
    index = []
    
    for i in range(iterations):
        # Perform forward and backward propagation
        cost, gradients, bias = costFunction(theta, b, X_train, y_train)
        
        # Update parameters using gradient descent
        theta = theta - alpha * gradients
        b = b - alpha * bias
        
        costs.append(cost)
        
        # Print cost every 100 iterations for monitoring
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print(f"Cost after iteration {i}: {cost}")
    
    parameters = {"theta": theta, "bias": b}
    plt.plot(index, cost_list2)
    plt.title("Cost-Iteration Relation")
    plt.xticks(index, rotation = "vertical")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
    
    return theta, b, costs


def initialize_weights_and_bias(dimension):
    theta = np.full((dimension, 1),0.01)
    b= 0.0
    return theta,b

def logistic_regression_encrypted(X_train, y_train):
    alpha = .001
    iterations = 4500
    theta, b, costs = grad_descent(X_train, y_train, alpha, iterations)


def main():
    #plaintext training
    X = prep.preprocess_data()[0] #(13500, 10), x_train
    y = prep.preprocess_data()[1] #13500, y_train
    X_test = prep.preprocess_data()[2]
    y_test = prep.preprocess_data()[3]
    
    alpha = .001
    iterations = 4500
    theta, b, costs = grad_descent(X, y, alpha, iterations)

    #get plaintext predictions
    z = np.dot(X_test, theta)+b
    h = sigmoid(z)
    predicted_y = np.zeros((1, X_test.shape[1]))
    predicted_y = (h >= 0.5).astype(int)
    #get plaintext accuracy
    accuracy = np.mean(y_test == predicted_y)
    error_rate = np.mean(y_test != predicted_y)

    print("Accuracy: ", accuracy)
    print("Error rate:", error_rate)
