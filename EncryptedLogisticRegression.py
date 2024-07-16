import tenseal as ts
import logisticRegression as lreg
import numpy as np
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
import preprocessData as prep


class EncryptedLogRegression:
    def __init__(self, context, x_train, y_train, alpha=.001, iterations=4500):
        self.context = context
        self.x_train = x_train  # encrypted
        self.y_train = y_train  # encrypted
        self.alpha = alpha
        self.iterations = iterations
        self.theta, self.b = self.initialize_weights_and_bias(self.context, self.x_train)
        self.gradient = ts.ckks_vector(context, [0.0] * len(x_train))
        self.gradient_b = ts.ckks_vector(context, [0.0])

    def initialize_weights_and_bias(self, context, x_train):
        """Initialize encrypted versions of theta and b """
        dimension = len(x_train)
        encrypted_theta = ts.ckks_vector(context, [0.0]*dimension)
        encrypted_b = ts.ckks_vector(context, [0.0])
        return encrypted_theta, encrypted_b

    def cost(self):
        m = len(self.y_train)
        ####forward propagation###
        # calculate z
        print(len(self.theta), len(self.b), len(self.x_train))
        z = [self.x_train[i].dot(self.theta) + self.b for i in range(len(self.x_train))]
        # calculate the hypothesis H0(x), predicted probability y = 1 (diabetic)
        h = [self.sigmoid(z_i) for z_i in z]

        cost = 0.0
        for i in range(m):
            y_i = self.y_train[i]
            h_i = h[i]
            # compute cost
            if y_i == 1:
                cost += (-y_i * h_i.log())
            else:
                cost += (-(1 - y_i) * (1 - h_i).log())
        cost = cost/m

        ### back propagation ###
        self.gradient = ts.ckks_vector(self.context, [0.0] * len(self.theta))
        self.gradient_b = ts.ckks_vector(self.context, [0.0])

        for i in range(m):
            error = h[i] - self.y_train[i]
            self.gradient += self.x_train[i] * error
            self.gradient_b += error
        self.gradient /= m
        self.gradient_b /= m

        return cost

    def grad_descent(self):
        costs = []
        cost_list2 = []
        index = []

        for i in range(self.iterations):
            cost = self.cost()
            costs.append(cost)

            # Update parameters using gradient descent
            self.theta -= self.alpha * self.gradient
            self.b -= self.alpha * self.gradient_b

            # Print cost every 100 iterations for monitoring
            if i % 100 == 0:
                cost_list2.append(cost.decrypt())
                index.append(i)
                print(f"Cost after iteration {i}: {cost.decrypt()}")

        # Plot cost vs iterations
        plt.plot(index, cost_list2)
        plt.title("Cost-Iteration Relation")
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.show()

    def sigmoid(z):
        """ Need a polynomial approximation to use encrypted data.
        Exponential functions can not be directly applied to 
        encrypted data.
        This function will use polynomial approximation by Chebyshev
        use a third order polynomial and [-6, 6] range to capture sigmoid transition from 0 to 1
        """
        # coefficients = Chebyshev.fit(np.linspace(-6, 6, 1000), 0.5 + 0.197 * np.linspace(-6, 6, 1000) - 0.004 * np.linspace(-6, 6, 1000) ** 3, 3).convert().coef
        coefficients = Chebyshev.fit(
            np.linspace(-6, 6, 1000), lreg.sigmoid(np.linspace(-6, 6, 1000)), 3).convert().coef
        h = coefficients[0] + coefficients[1] * z + \
            coefficients[2] * z**2 + coefficients[3] * z**3
        return h

    def get_predictions(self, x_test):
        # Decrypt theta and b for evaluation
        theta_plain = self.theta.decrypt()
        b_plain = self.b.decrypt()

        # Calculate predictions
        z = np.dot(x_test, theta_plain) + b_plain
        h = self.sigmoid(z)
        predicted_y = (h >= 0.5).astype(int)

        # Encrypt the predictions for privacy-preserving evaluation
        encrypted_predicted_y = [ts.ckks_vector(
            self.context, [y]) for y in predicted_y]
        return encrypted_predicted_y

    def get_evaluation(self, encrypted_predicted_y, y_test):
        # Decrypt predicted values for evaluation
        predicted_y = np.array([y.decrypt() for y in encrypted_predicted_y])

        # Decrypt y_test for evaluation
        y_test_plain = y_test.decrypt()

        # Calculate evaluation metrics
        accuracy = np.mean(y_test_plain == predicted_y)
        error_rate = np.mean(y_test_plain != predicted_y)

        print("Accuracy: ", accuracy)
        print("Error rate:", error_rate)

        return accuracy, error_rate


def main():
    # get data
    ##### MOVE DATA ENcryption to diff file#########
    data = prep.preprocess_data()
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]

    # CKKS tenseal parameters
    poly_mod_degree = 32768
    deg_of_optimization = -1
    # coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    coeff_mod_bit_sizes = [60, 40, 60]
    # create a tenseal context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree,
                         deg_of_optimization, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 21
    context.generate_galois_keys()

    # encrypt training data using ckks (need tolist() not numpy so compatible with tenseal)
    encrypted_x_train = [ts.ckks_vector(context, x.tolist()) for x in x_train]
    encrypted_y_train = [ts.ckks_vector(context, [y]) for y in y_train.tolist()]

    # #initialize model
    model = EncryptedLogRegression(context, encrypted_x_train, encrypted_y_train)

    # train model
    model.grad_descent()
    print("Model training finished!")

    # #get plaintext predictions
    y_predicted = model.get_predictions(x_test)

    # #get plaintext accuracy
    model.get_evaluation(y_predicted, y_test)


main()