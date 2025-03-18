import numpy as np

def create_montage(blocks, h, w, block_size):
    """ Create a montage of blocks and ensure the final size is the same as the original image """
    montage_image = np.zeros((h, w))
    block_idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = blocks[block_idx]
            # Ensure block fits inside the image boundaries
            block_height = min(block_size, h - i)
            block_width = min(block_size, w - j)
            montage_image[i:i+block_height, j:j+block_width] = block[:block_height, :block_width]
            block_idx += 1
    return montage_image