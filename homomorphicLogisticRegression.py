import homomorphicEngryption as HE
import preprocessData as prep
import logisticRegression as lreg
import numpy as np
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
import tenseal as ts


def encrypt_data(ckks):
    # encrypt patient_data
    X_train, y_train, X_test, y_test = prep.preprocess_data()
    encoded_x_train = ckks.encode(X_train)
    encoded_y_train = ckks.encode(y_train)
    encoded_x_test = ckks.encode(X_test)
    encoded_y_test = ckks.encode(y_test)

    encrypted_x_train = ckks.encrypt(encoded_x_train)
    encrypted_y_train = ckks.encrypt(encoded_y_train)
    encrypted_x_test = ckks.encrypt(encoded_x_test)
    encrypted_y_test = ckks.encrypt(encoded_y_test)

    return encrypted_x_train, encrypted_y_train, encrypted_x_test, encrypted_y_test


def encrypt_logistic_parameters(ckks, encrypted_x_train):
    # ecnrypt parameters needed for logistic regression
    theta, b = lreg.initialize_weights_and_bias(encrypted_x_train.shape[1])
    encoded_theta = ckks.encode(theta) 
    encoded_b = ckks.encode(b)
    encrypted_theta = ckks.encrypt(encoded_theta)
    encrypted_b = ckks.encrypt(encoded_b)
    return encrypted_theta, encrypted_b


def sigmoid_encrypted_data(z):
    """ Need a polynomial approximation to use encrypted data.
        Exponential functions can not be directly applied to 
        encrypted data.
        This function will use polynomial approximation by Chebyshev
    """
    # use a third order polynomial and [-6, 6] range to capture sigmoid transition from 0 to 1
    coefficients = Chebyshev.fit(np.linspace(-6, 6, 1000), lreg.sigmoid(np.linspace(-6, 6, 1000)), 3).convert().coef
    h = coefficients[0] + coefficients[1] * z + coefficients[2] * z**2 + coefficients[3] * z**3
    return h


def costFunction_encrypted_data(theta, b, X_train, y_train, ckks):
    """ Need a polynomial approximation to use encrypted data.
        Logarithmic functions can not be directly applied to 
        encrypted data.
        This function will use polynomial approximation by Chebyshev
    """
    m = len(X_train)
    ####forward propagation###
    # CKKS supports element-wise multiplication of encrypted data
    z = X_train @ theta + b
    h = sigmoid_encrypted_data(z)
    # y_train = y_train.reshape(-1, 1)

    # polynomial approximation
    diff = h - y_train
    cost = np.matmul(diff, diff) / (2 * m)

    ###back propagation###
    bias = np.sum(diff) / m
    bias = ts.CKKSVector(ckks, bias)
    gradient = np.matmul(X_train.transpose(), diff) / m
    return cost, gradient, bias


def grad_descent_encrypted_data(X_train, y_train, b, theta, ckks, alpha, iterations):
    m = X_train.shape[0]  # Number of training examples

    # Initialize weights and bias
    costs = []
    cost_list2 = []
    index = []

    enc_alpha = ckks.encode(alpha)  # Encrypt alpha for use in gradient descent

    for i in range(iterations):
        # Perform forward and backward propagation
        cost, gradients, bias = costFunction_encrypted_data(theta, b, X_train, y_train, ckks)

        # Update parameters using gradient descent (encrypted operations)
        theta = theta.sub(enc_alpha.mul(gradients))
        b = b.sub(enc_alpha.mul(bias))

        costs.append(cost)

        # Print cost every 100 iterations for monitoring
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print(f"Cost after iteration {i}: {cost}")

    # Plotting cost versus iterations
    plt.plot(index, cost_list2)
    plt.title("Cost-Iteration Relation")
    plt.xticks(index, rotation="vertical")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()

    return theta, b, costs


def main():
    # tests
    # params = CKKSParameters(poly_modulus_degree=8, coeff_modulus_bits=[60, 40, 60])
    # params_16384 = CKKSParameters(poly_modulus_degree=16384, coeff_modulus_bits=[60, 40, 60])
    # params_32768 = CKKSParameters(poly_modulus_degree=32768, coeff_modulus_bits=[60, 40, 60])
    # tests
    # params_40 = CKKSParameters(poly_modulus_degree=8192, coeff_modulus_bits=[40, 30, 30])
    # params_60 = CKKSParameters(poly_modulus_degree=8192, coeff_modulus_bits=[60, 40, 20])
    # params_80 = CKKSParameters(poly_modulus_degree=8192, coeff_modulus_bits=[80, 60, 40])

    params_8192 = HE.CKKSParameters(poly_modulus_degree=8192, coeff_modulus_bits=[60, 40, 60])
    ckks = HE.CKKS(params_8192)
    encrypted_x_train, encrypted_y_train, encrypted_x_test, encrypted_y_test = encrypt_data(ckks)
    encrypted_theta, encrypted_b = encrypt_logistic_parameters(ckks, encrypted_x_train)

    grad_descent_encrypted_data(encrypted_x_train, encrypted_y_train, encrypted_b, encrypted_theta, ckks, alpha=.001, iterations=4500)




if __name__ == '__main__':
    main()
