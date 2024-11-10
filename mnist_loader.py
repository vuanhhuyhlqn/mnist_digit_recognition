import pickle
import gzip

import numpy as np

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, test_data)

def load_data_wrapper():
    tr, te = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr[0]]
    training_results = [vectorized_result(y) for y in tr[1]]
    training_data = list(zip(training_inputs, training_results))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te[0]]
    test_data = list(zip(test_inputs, te[1]))
    
    return (training_data, test_data)

def vectorized_result(num):
    e = np.zeros((10, 1))
    e[num] = 1.0
    return e
