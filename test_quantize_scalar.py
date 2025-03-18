import numpy as np

def test_quantize_scalar(value, codebook):
    """
    Given a single float 'value' and a 1D array 'codebook',
    return the nearest codebook entry.
    """
    # Find the index of the nearest codebook value
    idx = np.argmin((codebook - value)**2)
    return codebook[idx]