import numpy as np

def normalize_image(image):
    """ Normalize the image for better visualization """
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)  # Scale to 0-1 range