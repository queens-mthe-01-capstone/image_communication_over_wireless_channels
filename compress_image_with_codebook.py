from blockify_image import blockify_image
from quantize_block_dct import quantize_block_dct
import numpy as np

def compress_image_with_codebook(image_array, dc_codebook, ac_codebook, block_size=8):
    """
    1) Break the image into blocks of size 'block_size x block_size'.
    2) For each block, do DCT -> quantize with codebook
    3) return the string of indexes
    """
    # Break image into blocks
    blocks = blockify_image(image_array, block_size)
    h, w = image_array.shape

    # Compress each block with DCT + codebook quantization
    compressed_block_indexes = []
    for block in blocks:
        # Convert block to float if not already
        block_float = block.astype(np.float32)
        # Force block into [−4,4] range or scale as needed if codebook was trained in that range
        block_float = (block_float / 255.0) * 8 - 4

        block_flat, block_indexes_flat = quantize_block_dct(block_float, dc_codebook, ac_codebook)

        compressed_block_indexes.append(block_indexes_flat)

    return compressed_block_indexes