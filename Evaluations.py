from BootstrapEncLogReg import BootstrappedEncryptedLogRegression
from EncryptedLogisticRegression import EncryptedLogRegression
from logisticRegression import LogisticRegression
import preprocessData as prep
import tenseal as ts
import numpy as np


### TODO ###
"""
Print graphs of cost over iterations
Create table of accuracies and error rates with parameters (iterations, alpha)
Evaluate entropy of encryption
Evaluate privacy
Maybe edit preprocess file to handle the encryption of data as well
Look through code and make aesthetic changes
"""