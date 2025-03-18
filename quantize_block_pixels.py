import numpy as np
from test_quantize_scalar import test_quantize_scalar

def quantize_block_pixels(block, codebook):
    block_quant = np.zeros_like(block, dtype=float)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            block_quant[i, j] = test_quantize_scalar(block[i, j], codebook)
    return block_quant