from test_quantize_scalar import test_quantize_scalar
from dct import DCT
import numpy as np

def test_quantize_block_dct(block, dc_codebook, ac_codebook):
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
    dc_encoded = test_quantize_scalar(dc_value, dc_codebook)
    ac_encoded = [test_quantize_scalar(a, ac_codebook) for a in ac_values]


    # 4) Rebuild the DCT block to do an IDCT
    #    (Alternatively you can skip if you only want to store encoded coefficients.)
    reconstructed_flat = np.array([dc_encoded] + ac_encoded)
    block_dct_reconstructed = reconstructed_flat.reshape(block.shape)

    block_idct = DCT(block_dct_reconstructed, block_size).inv_dct_block()

    # Return the (decoded) 2D block
    return block_idct