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
        return np.fft.fft(values) * self.scale

    def decode(self, encoded_values):
        return np.real(np.fft.ifft(encoded_values / self.scale))

    def encrypt(self, plaintext):
        plaintext_shape = plaintext.shape
        # Truncate or pad noise to match plaintext rows
        if self.noise.shape[0] < plaintext_shape[0]:
            # Pad noise with zeros if necessary
            noise_adjusted = np.pad(self.noise, (0, plaintext_shape[0] - self.noise.shape[0]), mode='constant')
        else:
            # Truncate noise if it has more elements than plaintext rows
            noise_adjusted = self.noise[:plaintext_shape[0]]
        noise_adjusted = noise_adjusted.reshape(-1, 1)

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