from vector_quantization.vector_test.decoder import *
from vector_quantization.vector_test.encoder import *
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# --- Defaults ---
DEFAULT_K = 16
DEFAULT_N = 256
DEFAULT_EPSILON = 0.1
DEFAULT_IMAGE = 'satelliteIMG02.jpg'

# --- Valid Options ---
VALID_K = [4, 16, 64]
VALID_N = [4,8,16,32,64,128,256]
VALID_EPSILON = [0,0.005,0.01,0.05,0.1]


vector_root = Path(__file__).resolve().parent.parent
project_root = vector_root.parent

image_path = str(project_root / 'images/test/misc_images')

covq_codebooks_path = str(vector_root / 'vector_codebooks/covq')
vq_codebooks_path = str(vector_root / 'vector_codebooks/vq')


parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, help='Vector dimension K (4, 16, or 64)')
parser.add_argument('--n', type=int, help='Number of codewords N (Powers of 2 from 2^2 to 2^8)')
parser.add_argument('--epsilon', help='Noise epsilon (0, 0.005, 0.01, 0.05, or 0.1)')
parser.add_argument('--image', help='Image filename (from images/test/misc_images/)')

args = parser.parse_args()

# --- Validation with Fallbacks ---
if args.k in VALID_K:
    dimension = args.k
else:
    dimension = DEFAULT_K
    if args.k is not None:
        print(f"[INFO] Invalid dimension '{args.k}', defaulting to {DEFAULT_K}")

if args.n in VALID_N:
    num_codewords = args.n
else:
    num_codewords = DEFAULT_N
    if args.n is not None:
        print(f"[INFO] Invalid number of codewords '{args.n}', defaulting to {DEFAULT_N}")

if args.epsilon in VALID_EPSILON:
    epsilon = args.epsilon
else:
    epsilon = DEFAULT_EPSILON
    if args.epsilon is not None:
        print(f"[INFO] Invalid epsilon '{args.epsilon}', defaulting to {DEFAULT_EPSILON}")

# Use image filename or default
image_filename = args.image if args.image else DEFAULT_IMAGE
test_img = image_path + "/" + image_filename

print("Running COVQ and VQ on " + test_img + " using the following parameters: \nDimension: " + str(dimension) + "\nnum_codewords: " + str(num_codewords) + "\nepsilon: " + str(epsilon))
print("\n")

delta = 10 # covq was trained with delta = 10
rate = int(math.log2(num_codewords))
block_size = int(math.sqrt(dimension))

if epsilon == 0.005:
    epsilonIdx = 1
    R=5
    B=995
elif epsilon == 0:
    epsilonIdx = 0
    R=0
    B=1
elif epsilon == 0.05:
    epsilonIdx = 3
    R=5
    B=95
elif epsilon == 0.1:
    epsilonIdx = 2
    R=1
    B=9
else: # sets epsilon to 0.1
    epsilonIdx = 4
    R=1
    B=99

G=0
M = 1


grey_img = grey_image_array(test_img)
covq_codebooks = np.load(covq_codebooks_path + "/codebooks_" + str(dimension) + "dimension_"+str(num_codewords)+"codewords.npz")
vq_codebooks = np.load(vq_codebooks_path + "/codebooks_" + str(dimension) + "dimension.npz")

covq_grey_img_norm = (grey_img - covq_codebooks["mean"]) / covq_codebooks["std"]
vq_grey_img_norm = (grey_img - vq_codebooks["mean"]) / vq_codebooks["std"]

init_b = list(range(num_codewords))

# get trained codebook for covq and vq
covq_codebook_string = "epsilon"+str(epsilonIdx)
covq_codebook = covq_codebooks[covq_codebook_string]

vq_codebook_string = "rate" + str(rate)
vq_codebook = vq_codebooks[vq_codebook_string]

# encode image and convert to bits
covq_indices = encode_image_blocks(covq_grey_img_norm, covq_codebook, init_b, epsilon, delta, block_size)
covq_bitstream = indices_to_bitstream(covq_indices, rate)

vq_indices = encode_image_blocks(vq_grey_img_norm, vq_codebook, init_b, epsilon, delta, block_size)
vq_bitstream = indices_to_bitstream(vq_indices, rate)

# put indices through polya channel and convert back to indices
covq_channel_bitstream = implement_channel(covq_bitstream, R, G, B, M, delta)
covq_received_indices = bitstream_to_indices(covq_channel_bitstream, rate)

vq_channel_bitstream = implement_channel(vq_bitstream, R, G, B, M, delta)
vq_received_indices = bitstream_to_indices(vq_channel_bitstream, rate)

# reconstruct image
covq_reconstructed_img = decode_image_blocks(covq_received_indices, covq_codebook, block_size, grey_img.shape)
covq_reconstructed_img = covq_reconstructed_img * covq_codebooks["std"] + covq_codebooks["mean"]

vq_reconstructed_img = decode_image_blocks(vq_received_indices, vq_codebook, block_size, grey_img.shape)
vq_reconstructed_img = vq_reconstructed_img * vq_codebooks["std"] + vq_codebooks["mean"]


# --- Calculate PSNR for COVQ Reconstruction ---
psnrCOVQ = psnr(grey_img, covq_reconstructed_img)
print(f"COVQ PSNR: {psnrCOVQ:.2f} dB")

# --- Calculate PSNR for VQ Reconstruction ---
psnrVQ = psnr(grey_img, vq_reconstructed_img)
print(f"VQ PSNR: {psnrVQ:.2f} dB")

# --- Plot 3 Images in one figure ---
plt.figure(figsize=(14, 6))

# Original
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(grey_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

# COSQ Reconstructed
plt.subplot(1, 3, 2)
plt.title(f"COVQ Reconstructed\nPSNR={psnrCOVQ:.2f} dB")
plt.imshow(covq_reconstructed_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

# Lloyd–Max Reconstructed
plt.subplot(1, 3, 3)
plt.title(f"VQ Reconstructed\nPSNR={psnrVQ:.2f} dB")
plt.imshow(vq_reconstructed_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()