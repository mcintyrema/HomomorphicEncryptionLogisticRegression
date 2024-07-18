import tenseal as ts
import preprocessData as prep
from EncryptedLogisticRegression import EncryptedLogRegression

# serialize CKKSVector
def serialize(ckks_vector):
    return ckks_vector.serialize()

# deserialize CKKSVector 
def deserialize(context, serialized_ckks_vector):
        return ts.ckks_vector_from(context, serialized_ckks_vector)

# save encrypted data
def save_encrypted_data(filename, context, x_train, y_train, x_test, y_test):
    context_bytes = context.serialize(save_secret_key=True)
    with open(filename + '_context', 'wb') as f:
        f.write(context_bytes)
    
    with open(filename + '_x_train', 'wb') as f:
        for ckks_vector in x_train:
            ckks_vector_bytes = serialize(ckks_vector)
            f.write(len(ckks_vector_bytes).to_bytes(4, 'big'))
            f.write(ckks_vector_bytes)
    
    with open(filename + '_y_train', 'wb') as f:
        for ckks_vector in y_train:
            ckks_vector_bytes = serialize(ckks_vector)
            f.write(len(ckks_vector_bytes).to_bytes(4, 'big'))
            f.write(ckks_vector_bytes)
    
    with open(filename + '_x_test', 'wb') as f:
        for ckks_vector in x_test:
            ckks_vector_bytes = serialize(ckks_vector)
            f.write(len(ckks_vector_bytes).to_bytes(4, 'big'))
            f.write(ckks_vector_bytes)
    
    with open(filename + '_y_test', 'wb') as f:
        for ckks_vector in y_test:
            ckks_vector_bytes = serialize(ckks_vector)
            f.write(len(ckks_vector_bytes).to_bytes(4, 'big'))
            f.write(ckks_vector_bytes)

#  load encrypted data
def load_encrypted_data(filename):
    with open(filename + '_context', 'rb') as f:
        context_bytes = f.read()
    context = ts.context_from(context_bytes)
    
    def load_vectors(file_suffix):
        vectors = []
        with open(filename + file_suffix, 'rb') as f:
            while True:
                len_bytes = f.read(4)
                if not len_bytes:
                    break
                vec_length = int.from_bytes(len_bytes, 'big')
                ckks_vector_bytes = f.read(vec_length)
                try:
                    vectors.append(deserialize(context, ckks_vector_bytes))
                except MemoryError as e:
                    print(f"MemoryError: bad allocation while loading vector from {file_suffix}")
                    raise e
                except Exception as e:
                    print(f"Error while loading vector from {file_suffix}: {e}")
                    raise e
        return vectors

    x_train = load_vectors('_x_train')
    y_train = load_vectors('_y_train')
    x_test = load_vectors('_x_test')
    y_test = load_vectors('_y_test')
    
    return context, x_train, y_train, x_test, y_test

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
    encrypted_x_train = [ts.ckks_vector(context, x.tolist()) for x in x_train_sample]
    encrypted_y_train = [ts.ckks_vector(context, [y]) for y in y_train_sample.tolist()]
    encrypted_x_test = [ts.ckks_vector(context, x.tolist()) for x in x_test_sample]
    encrypted_y_test = [ts.ckks_vector(context, [y]) for y in y_test_sample.tolist()]
    
    # data handling
    # save_encrypted_data('encrypted_data_sample', context, encrypted_x_train, encrypted_y_train, encrypted_x_test, encrypted_y_test)

    # load data
    context, x_train_deserialized, y_train_deserialized, x_test_deserialized, y_test_deserialized = load_encrypted_data('encrypted_data_sample')
    print("Encrypted data loaded successfully.")
   
    # train model
    model = EncryptedLogRegression(context, x_train_deserialized, y_train_deserialized, x_test_deserialized, y_test_deserialized)
    model.grad_descent()
    print("Model training finished!")

    # Get predictions and evaluate model
    y_predicted = model.get_predictions()
    model.get_evaluation(y_predicted)

if __name__ == "__main__":
    main()
