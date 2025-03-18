import numpy as np

def blockify_image(image, block_size):
    """ Break the image into blocks of size block_size x block_size,
    handling edge cases where the image size is not divisible by block_size """
    blocks = []
    h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Extract block, ensuring it has the correct dimensions
            block = image[i:min(i + block_size, h), j:min(j + block_size, w)]

            # Pad the block if it's smaller than block_size x block_size
            if block.shape != (block_size, block_size):
                pad_height = block_size - block.shape[0]
                pad_width = block_size - block.shape[1]
                block = np.pad(block, ((0, pad_height), (0, pad_width)), 'edge')

            blocks.append(block)
    return blocks