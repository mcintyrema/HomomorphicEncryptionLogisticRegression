from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
import tenseal as ts
import numpy as np

class EncryptedLogRegression:
    def __init__(self, context, x_train, y_train, x_test, y_test, alpha=.5, iterations=700):
        """ Logistic Regression machine learning model for CKKS homomorphic
        encryption"""
        self.context = context
        self.x_train = x_train  # encrypted
        self.y_train = y_train  # encrypted
        self.x_test = x_test  # encrypted
        self.y_test = y_test  # encrypted
        self.n = len(x_train[0].decrypt()) # features
        self.m = len(y_train) #samples
        self.alpha = alpha
        self.iterations = iterations
        self.theta, self.b = self.initialize_weights_and_bias()
        self.gradient = ts.ckks_vector(context, [0.0] * self.n)
        self.gradient_b = ts.ckks_vector(context, [0.0])


    def initialize_weights_and_bias(self):
        """Initialize encrypted versions of theta and b """
        encrypted_theta = ts.ckks_vector(self.context, [0.0] * self.n)
        encrypted_b = ts.ckks_vector(self.context, [0.0])
        return encrypted_theta, encrypted_b


    def sigmoid(self, z):
        """ Need a polynomial approximation to use encrypted data.
        Exponential functions can not be directly applied to 
        encrypted data.
        This function will use polynomial approximation by Chebyshev
        use a third order polynomial and [-6, 6] range to capture sigmoid transition from 0 to 1
        """
        coefficients = Chebyshev.fit(np.linspace(-6, 6, 1000), sigmoid(np.linspace(-6, 6, 1000)), 3).convert().coef
        h = coefficients[0] + coefficients[1] * z + coefficients[2] * z**2 + coefficients[3] * z**3
        return h

    
    def cost(self):
        """Compute the cost by using the updated sigmoid function to 
        compute the hypothesis. Compare hypothesized values with ground
        truth y values. Compute the gradients to update the weights (theta) 
        and the bias value (b)"""

        ####forward propagation###
        theta_plain = self.theta.decrypt()
        b_plain = self.b.decrypt()[0]
        # calculate z
        z = [np.dot(self.x_train[i].decrypt(), theta_plain) + b_plain for i in range(len(self.x_train))]
        # calculate the hypothesis H0(x), predicted probability y = 1 (diabetic)
        h = [self.sigmoid(z_i) for z_i in z]

        cost = 0.0
        for i in range(self.m):
            y_i = self.y_train[i].decrypt()[0] 
            h_i = h[i]
            # compute cost
            cost += -y_i * np.log(h_i) - (1 - y_i) * np.log(1 - h_i)
        cost = cost * (1 / self.m)

        ### back propagation ###
        self.gradient = ts.ckks_vector(self.context, [0.0] * self.n)
        self.gradient_b = ts.ckks_vector(self.context, [0.0])

        for i in range(self.m):
            error = h[i] - self.y_train[i].decrypt()[0] 
            error_vector = [error] * self.n  
            self.gradient = self.gradient + self.x_train[i].mul(error_vector)
            self.gradient_b = self.gradient_b + ts.ckks_vector(self.context, [error])
        self.gradient = self.gradient * (1/self.m)
        self.gradient_b = self.gradient_b * (1/self.m)

        return cost
    

    def grad_descent(self):
        """Perform gradient descent by updating the weights and bias
        after each iteration by the gradients computed in the cost
        function. Plot the cost value vs iteration number to ensure the 
        model is learning and cost is decreasing. Printing included for
        development testing."""

        costs = []
        cost_list2 = []
        index = []

        for i in range(self.iterations):
            cost = self.cost()
            costs.append(cost)

            # update parameters
            self.theta -= self.alpha * self.gradient
            self.b -= self.alpha * self.gradient_b

            if i % 1 == 0:
                cost_list2.append(cost)
                index.append(i)
                print(f"Cost after iteration {i}: {cost}")

        # plots
        plt.plot(index, cost_list2)
        plt.title("Cost-Iteration Relation")
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.show()


    def get_predictions(self):
        """Decrypt the weights and bias and compute the predicted 
        values using test values and sigmoid function"""

        theta_plain = self.theta.decrypt()
        b_plain = self.b.decrypt()[0]

        # calculate predictions using polynomial approximation
        z = [np.dot(self.x_test[i].decrypt(), theta_plain) + b_plain for i in range(len(self.x_test))]
        z = np.array(z)  
        h = self.sigmoid(z)

        print(f"z: {z[:9]}")  
        print(f"h: {h[:9]}")  

        predicted_y = (h >= 0.5).astype(int)
        print(f"predicted_y: {predicted_y[:9]}") 
        encrypted_predicted_y = [ts.ckks_vector(self.context, [float(y)]) for y in predicted_y]
        
        return encrypted_predicted_y


    def get_evaluation(self, encrypted_predicted_y):
        """Evaluate the accuracy and error rate of the model using
        the predicted y values from the trained model and the ground
        truth values from the test set"""

        predicted_y = np.array([float(y.decrypt()[0]) for y in encrypted_predicted_y])
        y_test_plain = np.array([float(y.decrypt()[0]) for y in self.y_test])
        
        # check values are 0 or 1
        predicted_y = (predicted_y >= 0.5).astype(int)
        y_test_plain = (y_test_plain >= 0.5).astype(int)
        print(f"predicted_y: {predicted_y[:9]}")  
        print(f"Y_test: {y_test_plain[:9]}")
        
        accuracy = np.mean(y_test_plain == predicted_y)
        error_rate = np.mean(y_test_plain != predicted_y)

        print("Accuracy: ", accuracy)
        print("Error rate:", error_rate)

        return accuracy, error_rate
    
def sigmoid(z):
    """define sigmoid function for Chebyshev approximation in 
    update sigmoid function"""
    return 1 / (1 + np.exp(-z))

def plot_sigmoid_approximation():
    """Plot both the Chebychev polynomial approximation
    and original sigmoid function to ensure approzimation is accurate."""

    z_values = np.linspace(-6, 6, 1000)
    actual_sigmoid = 1 / (1 + np.exp(-z_values))
    cheb = Chebyshev.fit(z_values, actual_sigmoid, 3)
    approx_sigmoid = cheb(z_values)

    plt.plot(z_values, actual_sigmoid, label='Actual Sigmoid')
    plt.plot(z_values, approx_sigmoid, label='Chebyshev Approximation', linestyle='--')
    plt.legend()
    plt.title('Sigmoid vs. Chebyshev Polynomial Approx')
    plt.show()