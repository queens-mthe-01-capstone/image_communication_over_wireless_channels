import numpy as np

def generate_binary_array(num_centroids):
    k = int(np.log2(num_centroids))
    b = np.array([list(map(int, np.binary_repr(i, width=k))) for i in range(num_centroids)])
    return b