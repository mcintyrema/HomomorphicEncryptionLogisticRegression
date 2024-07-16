import tenseal as ts
import numpy as np
from EncryptedLogisticRegression import EncryptedLogRegression
import preprocessData as prep

class BootstrappedEncryptedLogRegression(EncryptedLogRegression):
    def __init__(self, context, x_train, y_train, alpha=0.001, iterations=4500, num_bootstraps=10):
        super().__init__(context, x_train, y_train, alpha, iterations)
        self.num_bootstraps = num_bootstraps
        self.bootstrap_samples = self.generate_bootstrap_samples()

    def generate_bootstrap_samples(self):
        m = len(self.y_train)
        bootstrap_samples = []
        for _ in range(self.num_bootstraps):
            indices = np.random.choice(range(m), size=m, replace=True)
            x_bootstrap = [self.x_train[i] for i in indices]
            y_bootstrap = [self.y_train[i] for i in indices]
            bootstrap_samples.append((x_bootstrap, y_bootstrap))
        return bootstrap_samples

    def train_with_bootstrapping(self):
        for idx, (x_bootstrap, y_bootstrap) in enumerate(self.bootstrap_samples):
            print(f"Training Bootstrap {idx+1}/{self.num_bootstraps}")
            self.x_train = x_bootstrap
            self.y_train = y_bootstrap
            self.theta, self.b = self.initialize_weights_and_bias()
            self.grad_descent()

def main():
    data = prep.preprocess_data()
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]

    poly_mod_degree = 32768
    deg_of_optimization = -1
    coeff_mod_bit_sizes = [60, 40, 60]

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, deg_of_optimization, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 21
    context.generate_galois_keys()

    encrypted_x_train = [ts.ckks_vector(context, x.tolist()) for x in x_train]
    encrypted_y_train = [ts.ckks_vector(context, [y]) for y in y_train.tolist()]

    model = BootstrappedEncryptedLogRegression(context, encrypted_x_train, encrypted_y_train)
    model.train_with_bootstrapping()
    print("Model training finished!")

    y_predicted = model.get_predictions(x_test)
    model.get_evaluation(y_predicted, y_test)

main()