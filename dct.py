import numpy as np

class DCT:
    """ This class can be used for both the DCT and inverse DCT """

    def __init__(self, signal, N):
        self.signal = signal  # 2-dimensional array
        self.N = N

    def alpha(self, u):
        """ DCT scaling factor """
        if u == 0:
            return np.sqrt(1 / self.N)
        else:
            return np.sqrt(2 / self.N)

    def c(self, u, v):
        """ Perform DCT on the signal using the given formula """
        constant = self.alpha(u) * self.alpha(v)
        temp_sum = 0
        for x in range(self.N):  # 0, 1,..., N-1
            for y in range(self.N):
                temp_sum += self.signal[x, y] * np.cos(((2*x + 1) * u * np.pi) / (2 * self.N)) * np.cos(((2*y + 1) * v * np.pi) / (2 * self.N))
        result = constant * temp_sum
        return result

    def f(self, x, y):
        """ Perform inverse DCT on the signal using the given formula """
        temp_sum = 0
        for u in range(self.N):  # 0, 1,..., N-1
            for v in range(self.N):
                temp_sum += self.alpha(u) * self.alpha(v) * self.signal[u, v] * np.cos(((2*x + 1) * u * np.pi) / (2 * self.N)) * np.cos(((2*y + 1) * v * np.pi) / (2 * self.N))
        return temp_sum

    def dct_block(self):
        """ Return the DCT of the input block """
        rows, cols = self.signal.shape
        signal_transform = np.zeros((rows, cols))
        for u in range(rows):
            for v in range(cols):
                signal_transform[u, v] = self.c(u, v)
        return signal_transform

    def inv_dct_block(self):
        """ Return the inverse DCT of the input block """
        rows, cols = self.signal.shape
        signal_transform = np.zeros((rows, cols))
        for x in range(rows):
            for y in range(cols):
                signal_transform[x, y] = self.f(x, y)
        return signal_transform