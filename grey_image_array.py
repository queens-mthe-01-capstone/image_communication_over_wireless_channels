from PIL import Image
import numpy as np

def grey_image_array(image_path):
  """
    import the image and store in a variable
    return the image pixel value array
  """
  raw_image = Image.open(image_path)
  grey_image = raw_image.convert("L")
  image_array = np.array(grey_image)
  return image_array