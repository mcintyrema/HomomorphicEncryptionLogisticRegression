import homomorphicEngryption as HE
import preprocessData as prep
import logisticRegression as lreg
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
    encypted_theta = ckks.encrypt(encoded_theta)
    encrypted_b = ckks.encrypt(encoded_b)



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
    encrypt_data(ckks)
