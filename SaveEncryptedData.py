"""For testing model only!!"""
import pickle
import tenseal as ts
import numpy as np
import preprocessData as prep
from EncryptedLogisticRegression import EncryptedLogRegression

def serialize_ckks_vector(vec):
    try:
        return vec.serialize()
    except Exception as e:
        print(f"Error serializing CKKSVector: {e}")
        return None

def deserialize_ckks_vector(context, serialized_vec):
    try:
        return ts.ckks_vector_from(context, serialized_vec)
    except Exception as e:
        print(f"Error deserializing CKKSVector: {e}")
        return None

def save_encrypted_data(filename, context, x_train, y_train, x_test, y_test):
    serialized_data = {
        'x_train': [serialize_ckks_vector(vec) for vec in x_train],
        'y_train': [serialize_ckks_vector(vec) for vec in y_train],
        'x_test': [serialize_ckks_vector(vec) for vec in x_test],
        'y_test': [serialize_ckks_vector(vec) for vec in y_test],
    }
    with open(filename, 'wb') as f:
        pickle.dump(serialized_data, f)
    context_bytes = context.serialize()
    with open(filename + '_context', 'wb') as f:
        f.write(context_bytes)


def load_encrypted_data(filename):
    with open(filename + '_context', 'rb') as f:
        context_bytes = f.read()
    context = ts.context_from(context_bytes)
    with open(filename, 'rb') as f:
        serialized_data = pickle.load(f)
    x_train = [deserialize_ckks_vector(context, vec) for vec in serialized_data['x_train']]
    y_train = [deserialize_ckks_vector(context, vec) for vec in serialized_data['y_train']]
    x_test = [deserialize_ckks_vector(context, vec) for vec in serialized_data['x_test']]
    y_test = [deserialize_ckks_vector(context, vec) for vec in serialized_data['y_test']]
    return context, x_train, y_train, x_test, y_test


data = prep.preprocess_data()
x_train = data[0]
y_train = data[1]
x_test = data[2]
y_test = data[3]

# CKKS tenseal parameters
poly_mod_degree = 32768
deg_of_optimization = -1
coeff_mod_bit_sizes = [60, 40, 60]
# create a tenseal context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, deg_of_optimization, coeff_mod_bit_sizes)
context.global_scale = 2 ** 21
context.generate_galois_keys()

encrypted_x_train = [ts.ckks_vector(context, x.tolist()) for x in x_train]
encrypted_y_train = [ts.ckks_vector(context, [y]) for y in y_train.tolist()]
encrypted_x_test = [ts.ckks_vector(context, x.tolist()) for x in x_test]
encrypted_y_test = [ts.ckks_vector(context, [y]) for y in y_test.tolist()]


save_encrypted_data('encrypted_data.pkl', context, encrypted_x_train, encrypted_y_train, encrypted_x_test, encrypted_y_test)
print("Encrypted data saved successfully.")
# Example usage:
context, x_train_deserialized, y_train_deserialized, x_test_deserialized, y_test_deserialized = load_encrypted_data('encrypted_data.pkl')
print("Encrypted data loaded successfully.")


model = EncryptedLogRegression(context, x_train_deserialized, y_train_deserialized)

# train model
model.grad_descent()
print("Model training finished!")

#get plaintext predictions
y_predicted = model.get_predictions(x_test_deserialized)

#get plaintext accuracy
model.get_evaluation(y_predicted, y_test_deserialized)
