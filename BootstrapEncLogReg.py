import tenseal as ts
import numpy as np
from EncryptedLogisticRegression import EncryptedLogRegression
import preprocessData as prep

class BootstrappedEncryptedLogRegression(EncryptedLogRegression):
    def __init__(self, context, x_train, y_train,x_test, y_test, alpha=0.5, iterations=700):
        super().__init__(context, x_train, y_train, x_test, y_test, alpha, iterations)
        self.strap_count = 7
        self.bootstrap_samples = self.generate_bootstrap_samples()

    def generate_bootstrap_samples(self):
        """Create samples to iteratively perform logisitic regression on"""
        m = len(self.y_train)
        bootstrap_samples = []
        for _ in range(self.strap_count):
            #sample with replacement
            indices = np.random.choice(range(m), size=m, replace=True)
            x_bootstrap = [self.x_train[i] for i in indices]
            y_bootstrap = [self.y_train[i] for i in indices]
            bootstrap_samples.append((x_bootstrap, y_bootstrap))
        return bootstrap_samples

    def train_with_bootstrapping(self):
        """ Use gradient descent from parent class to train model on 
        bootstrap samples"""

        for _, (x_bootstrap, y_bootstrap) in enumerate(self.bootstrap_samples):
            self.x_train = x_bootstrap
            self.y_train = y_bootstrap
            self.theta, self.b = self.initialize_weights_and_bias()
            self.grad_descent()


def main():
   # CKKS tenseal parameters
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    deg_of_optimization = -1
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, deg_of_optimization, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 21
    context.generate_galois_keys()
    
    # Preprocess data
    data = prep.preprocess_data()
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]

    # sampling
    sample_size = 100 
    y_samp = int(sample_size*.1)
    x_train_sample = x_train[:sample_size]
    y_train_sample = y_train[:sample_size]
    x_test_sample = x_test[:y_samp]
    y_test_sample = y_test[:y_samp]

    # encrypt data
    encrypted_x_train = [ts.ckks_vector(context, x.tolist(), scale=context.global_scale) for x in x_train_sample]
    encrypted_y_train = [ts.ckks_vector(context, [y], scale=context.global_scale) for y in y_train_sample.tolist()]
    encrypted_x_test = [ts.ckks_vector(context, x.tolist(), scale=context.global_scale) for x in x_test_sample]
    encrypted_y_test = [ts.ckks_vector(context, [y], scale=context.global_scale) for y in y_test_sample.tolist()]
    
    # train model
    model = BootstrappedEncryptedLogRegression(context, encrypted_x_train, encrypted_y_train, encrypted_x_test, encrypted_y_test)
    model.train_with_bootstrapping()
    print("Model training finished!")

    y_predicted = model.get_predictions()
    model.get_evaluation(y_predicted)