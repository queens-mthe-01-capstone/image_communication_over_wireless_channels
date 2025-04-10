import numpy as np
from PIL import Image


def grey_image_array(img):
  """
    import the image and store in a variable
    return the image pixel value array
  """
  raw_image = Image.open(img)
  grey_image = raw_image.convert("L").resize((512, 512))
  image_array = np.array(grey_image)
  return image_array


def blockify_image(img, block_size):
    """ Break the image into blocks of size block_size x block_size,
    handling edge cases where the image size is not divisible by block_size """
    blocks = []
    h, w = img.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Extract block, ensuring it has the correct dimensions
            block = img[i:min(i + block_size, h), j:min(j + block_size, w)]

            # Pad the block if it's smaller than block_size x block_size
            if block.shape != (block_size, block_size):
                pad_height = block_size - block.shape[0]
                pad_width = block_size - block.shape[1]
                block = np.pad(block, ((0, pad_height), (0, pad_width)), 'edge')

            blocks.append(block.ravel())
    return blocks

