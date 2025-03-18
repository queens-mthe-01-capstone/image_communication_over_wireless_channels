from quantize_scalar import quantize_scalar
import numpy as np
from dct import DCT

def quantize_block_dct(block, dc_codebook, ac_codebook):
    """
    1) Apply 2D DCT to 'block'.
    2) Separate DC & AC coefficients.
    3) Quantize (encode) them with dc_codebook & ac_codebook.
    4) Dequantize or decode them (optional), then do inverse DCT for reconstruction.
    5) Return the reconstructed block (2D array).
    """
    block_size = 8
    # 1) 2D DCT
    block_dct = DCT(block, block_size).dct_block()

    # 2) Flatten in either row-major or zigzag
    flattened = block_dct.flatten()
    dc_value = flattened[0]
    ac_values = flattened[1:]

    # 3) Quantize / encode
    dc_encoded, dc_idx  = quantize_scalar(dc_value, dc_codebook)
    # Fix: Separate the list of tuples into two lists
    ac_results = [quantize_scalar(a, ac_codebook) for a in ac_values]
    ac_encoded = [result[0] for result in ac_results]  # Extract encoded values
    ac_idx = [result[1] for result in ac_results]      # Extract indices


    # 4) Rebuild the DCT block to do an IDCT
    #    (Alternatively you can skip if you only want to store encoded coefficients.)
    block_flat = np.array([dc_encoded] + ac_encoded)
    block_indexes_flat = np.array([dc_idx] + ac_idx)

    # Return the flat block and return the index of the flat block
    return block_flat, block_indexes_flat