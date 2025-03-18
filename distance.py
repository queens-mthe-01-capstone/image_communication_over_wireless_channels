import numpy as np

# Distance part of the equation
def distance(y_i, y_j):
    dist = np.linalg.norm(y_i - y_j)
    dist = dist ** 2
    return dist