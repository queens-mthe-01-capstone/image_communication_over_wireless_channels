import numpy as np
from PIL import Image
from grey_image_array import grey_image_array
from crop_to_square import crop_to_square
from compress_image_with_codebook import compress_image_with_codebook
from print_compressed_array import print_compressed_array

loaded_dc_codebooks = np.load("trained_dc_codebooks.npz")
dc_codebooks = [loaded_dc_codebooks[f"arr_{i}"] for i in range(7)]

loaded_ac_codebooks = np.load("trained_ac_codebooks.npz")
ac_codebooks = [loaded_ac_codebooks[f"arr_{i}"] for i in range(7)]


image_path = 'images/Baboon.jpg'
raw_image = Image.open(image_path)


grey_image = grey_image_array(image_path)
image_array = crop_to_square(grey_image)

dc_codebook = dc_codebooks[0]
ac_codebook = ac_codebooks[0]

# 3) Compress image using codebook
compressed_img_array = compress_image_with_codebook(image_array, dc_codebook, ac_codebook, block_size=8)

print(f"Compressed Shape: {len(compressed_img_array)}")
# Call the function to print the compressed array
print_compressed_array(compressed_img_array)