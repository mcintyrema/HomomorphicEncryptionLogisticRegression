import matplotlib.pyplot as plt
import preprocessData as prep
import numpy as np


class LogisticRegression:
    def __init__(self, data, alpha=0.001, iterations=4500):
        self.X_train = data[0] #(13500, 10), x_train
        self.y_train = data[1] #13500, y_train
        self.x_test = data[2]
        self.y_test = data[3]
        self.alpha = alpha
        self.iterations = iterations
        self.theta, self.b = self.initialize_weights_and_bias(self.X_train.shape[1])
        
    def initialize_weights_and_bias(self, dimension):
        self.theta = np.full((dimension, 1),0.01)
        self.b= 0.0
        return self.theta, self.b

    def sigmoid(self, z):
        # calc exponential
        temp = (1+np.exp(-z))
        g = 1/temp
        return g

    def costFunction(self):
        m = len(self.X_train) # number of training samples

        ####forward propagation###
        # calculate z
        z = np.dot(self.X_train, self.theta) + self.b
        # calculate the hypothesis H0(x), predicted probability y = 1 (diabetic)
        h = self.sigmoid(z)  #(13500,)
        #stabilize data for logarithms
        self.y_train = self.y_train.reshape(-1, 1) #(13500,)
        epsilon = 1e-10
        #remove almost zero and 1 values from h
        h = np.clip(h, epsilon, 1 - epsilon)
        
        sum0 = -self.y_train*np.log(h)
        sum1 = sum0 - (1-self.y_train)*np.log(1-h)
        cost = (np.sum(sum1)) * (1/m)

        ###back propagation###
        gradient_b = np.sum(h - self.y_train) / m
        gradient = (1/m)*np.dot(self.X_train.T, (h-self.y_train))

        return cost, np.array(gradient), gradient_b

    def grad_descent(self):
        m = self.X_train.shape[0]  # Number of training examples
        
        # Initialize cost
        costs = []
        cost_list2 = []
        index = []
        
        for i in range(self.iterations):
            # Perform forward and backward propagation
            cost, gradients, gradient_b = self.costFunction()
            
            # Update parameters using gradient descent
            self.theta = self.theta - self.alpha * gradients
            self.b = self.b - self.alpha * gradient_b
            
            costs.append(cost)
            
            # Print cost every 100 iterations for monitoring
            if i % 100 == 0:
                cost_list2.append(cost)
                index.append(i)
                print(f"Cost after iteration {i}: {cost}")
        
        parameters = {"theta": self.theta, "bias": self.b}
        plt.plot(index, cost_list2)
        plt.title("Cost-Iteration Relation")
        plt.xticks(index, rotation = "vertical")
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.show()
        
        return self.theta, self.b, costs
    
    def get_predictions(self):
        z = np.dot(self.x_test, self.theta)+self.b
        h = self.sigmoid(z)
        predicted_y = np.zeros((1, self.x_test.shape[1]))
        predicted_y = (h >= 0.5).astype(int)
        return predicted_y

    def get_evaluation(self, predicted_y):
        accuracy = np.mean(self.y_test == predicted_y)
        error_rate = np.mean(self.y_test != predicted_y)

        print("Accuracy: ", accuracy)
        print("Error rate:", error_rate)
        return accuracy, error_rate


def main():
    #plaintext training
    patient_data = prep.preprocess_data()
    model = LogisticRegression(patient_data, .001, 4500)

    #train model
    model.grad_descent()

    #get plaintext predictions
    y_predicted = model.get_predictions()
    
    #get plaintext accuracy
    model.get_evaluation(y_predicted)