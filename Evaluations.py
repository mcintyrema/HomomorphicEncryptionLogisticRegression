from BootstrapEncLogReg import BootstrappedEncryptedLogRegression
from EncryptedLogisticRegression import EncryptedLogRegression
from logisticRegression import LogisticRegression
import EncryptedData as enc_data
import matplotlib.pyplot as plt
from collections import Counter
import preprocessData as prep
import tenseal as ts
import math


def create_accuracy_chart():
    ### Plaintext data ###
    patient_data = prep.preprocess_data()
    model0 = LogisticRegression(patient_data, .001, 4500)
    #train Algorithm 0
    model0.grad_descent()
    # evaluate Algorithm 0
    y_predicted = model0.get_predictions()
    plain_acc, plain_error = model0.get_evaluation(y_predicted)
    print("Plaintext done!")

    #### Encrypted data A ###
    context, x_train_deserialized, y_train_deserialized, x_test_deserialized, y_test_deserialized = enc_data.load_encrypted_data("encrypted_data_sample")
    # train Algorith A
    modelA = EncryptedLogRegression(context, x_train_deserialized, y_train_deserialized, x_test_deserialized, y_test_deserialized)
    modelA.grad_descent()
    #  evaluate model Algorith A
    y_predicted = modelA.get_predictions()
    enc_accuracy, enc_error = modelA.get_evaluation(y_predicted)
    print("A done!")

    #### Encrypted data B ###
    modelB = BootstrappedEncryptedLogRegression(context, x_train_deserialized, y_train_deserialized, x_test_deserialized, y_test_deserialized)
    modelB.train_with_bootstrapping()
    y_predicted = modelB.get_predictions()
    enc_accuracyB, enc_errorB = modelB.get_evaluation(y_predicted)
    print("boot done!")

    plain_alpha = model0.alpha
    encrypted_alpha = modelA.alpha
    encrypted_alphb = modelB.alpha
    plain_iterations = model0.iterations
    enc_iterations = modelA.iterations
    enc_iterations_b = modelB.iterations

    data = [
        ["Model", "Alpha", "Iteration Count", "Accuracy", "Error Rate", "Sample"],
        ["Algorithm 0", plain_alpha, plain_iterations, plain_acc, plain_error, patient_data[0].shape[0]],
        ["Algorithm A", encrypted_alpha, enc_iterations, enc_accuracy, enc_error, 100],
        ["Algorithm b", encrypted_alphb, enc_iterations_b, enc_accuracyB, enc_errorB, 100]
    ]

    # create table as png
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.tight_layout()
    plt.savefig("comparison_table.png", bbox_inches='tight', dpi=300)
    plt.show()


def calculate_entropy(data):
    """
    Find the Shannon entropy inputs.    
    """
    byte_counts = Counter(data)
    total_bytes = len(data)
    
    entropy = 0.0
    for count in byte_counts.values():
        probability = count / total_bytes
        entropy -= probability * math.log2(probability)
    print(entropy)
    return entropy


def main():
    create_accuracy_chart()
    patient_data = prep.preprocess_data()
    x_train0 = patient_data[0]
    y_train0 = patient_data[1]
    x_test0 = patient_data[2]
    y_test0 = patient_data[3] 

    # CKKS tenseal parameters
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    deg_of_optimization = -1
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, deg_of_optimization, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 21
    context.generate_galois_keys()

    # encrypted data
    x_train_A = [ts.ckks_vector(context, x.tolist()) for x in x_train0]
    y_train_A = [ts.ckks_vector(context, [y]) for y in y_train0.tolist()]
    x_test_A = [ts.ckks_vector(context, x) for x in x_test0]
    y_test_A = [ts.ckks_vector(context, [y]) for y in y_test0.tolist()]


    # get entropy
    entropy_test = [x_train0[:100].tolist(), y_train0[:100].tolist(), x_test0[:100].tolist(), y_test0[:100].tolist(), x_train_A, y_train_A, x_test_A, y_test_A]
    entropies = []
    for i in entropy_test:
        entropies.append(round(calculate_entropy(i), 2))

    data = [
        ["x_train0", "y_train0", "x_test0", "y_test0", "x_train_A", "y_train_A", "x_test_A", "y_test_A"],
        [entropies[0], entropies[1], entropies[2], entropies[3], entropies[4], entropies[5], entropies[6], entropies[7]],
       ]
    
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    plt.tight_layout()
    plt.savefig("entropy.png", bbox_inches='tight', dpi=300)
    plt.show()
main()