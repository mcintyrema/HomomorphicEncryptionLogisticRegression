import numpy as np

class CKKSParameters:
    def __init__(self, poly_modulus_degree, coeff_modulus_bits):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_modulus_bits = coeff_modulus_bits
        self.scale = 2 ** (max(coeff_modulus_bits) // 2)

class CKKS:
    def __init__(self, params):
        self.params = params
        self.poly_modulus_degree = params.poly_modulus_degree
        self.scale = params.scale
        self.secret_key = np.random.randint(2, size=self.poly_modulus_degree)
        self.noise = np.random.randn(self.poly_modulus_degree)


    def encode(self, values):
        #fourier transform input and scale (batch processing)
        values_complex = np.asarray(values, dtype=np.complex128)

        if np.iscomplexobj(values_complex) and values_complex.ndim:
            encoded_values = np.fft.fft(values_complex) * self.scale
        else:
            encoded_values = values_complex * self.scale
        return encoded_values


    def decode(self, encoded_values):
        return np.real(np.fft.ifft(encoded_values / self.scale))


    def encrypt(self, plaintext):
        # Convert plaintext to NumPy array if it's not already
        if not isinstance(plaintext, np.ndarray):
            plaintext = np.array(plaintext)

        # Determine plaintext shape and number of dimensions
        plaintext_shape = plaintext.shape
        num_dims = plaintext.ndim

        # Initialize noise_adjusted
        noise_adjusted = np.zeros_like(plaintext, dtype=np.float64)

        # Adjust noise to match plaintext shape
        if num_dims == 0:  # Scalar case
            noise_adjusted = self.noise[0]
        elif num_dims == 1:  # 1D array case
            if self.noise.shape[1] == 1:
                noise_adjusted[:plaintext_shape[0]] = self.noise[:plaintext_shape[0], 0]
            elif self.noise.shape[0] < plaintext_shape[0]:
                noise_adjusted[:self.noise.shape[0]] = self.noise[:, 0]
            else:
                noise_adjusted[:plaintext_shape[0]] = self.noise[:plaintext_shape[0], 0]
        elif num_dims == 2:  # 2D matrix case
            if self.noise.shape[0] < plaintext_shape[0]:
                noise_adjusted[:self.noise.shape[0], :] = self.noise[:, np.newaxis]
            else:
                noise_adjusted[:plaintext_shape[0], :] = self.noise[:plaintext_shape[0], np.newaxis]
        else:
            raise ValueError(f"Unsupported plaintext dimensionality: {num_dims}")
        # Perform addition
        try:
            ciphertext = plaintext + noise_adjusted
            print("Addition successful!")
            self.noise = noise_adjusted
            return ciphertext
        except ValueError as e:
            print("Error during addition:", e)



    def decrypt(self, ciphertext):
        # batch subtraction
        return ciphertext - self.noise