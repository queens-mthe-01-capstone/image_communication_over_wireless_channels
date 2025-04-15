import numpy as np
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
from random import randrange

#result of modulo 2 adding x1 and x2
def modulo_add(x1, x2):
  if x1 == x2:
    return 0
  else:
    return 1
  
def indices_to_bitstream(indices, rate):
    bitstream = []
    for idx in indices:
        # Convert index to a binary string with leading zeros to make up 'rate' bits
        bits = [int(b) for b in format(idx, '0{}b'.format(rate))]
        bitstream.extend(bits)
    return bitstream

def bitstream_to_indices(bitstream, rate):
    indices = []
    for i in range(0, len(bitstream), rate):
        bits = bitstream[i:i+rate]
        idx = int(''.join(str(b) for b in bits), 2)
        indices.append(idx)
    return np.array(indices)

def simulate_channel_string(len_string, R, G, B, M, Del, debug=False):
    """
    Simulate a bursty bit-flip channel via a Polya urn process.

    Internal "colors":
       - 'B' -> we record a 0
       - 'R' -> we record a 1
       - 'G' -> we also record a 1 (no 2 in the output)

    Args:
    -------
    len_string : int
        Number of picks (output bits) to generate.
    R, G, B : int
        Initial counts of Red, Green, and Blue in the urn.
    M : int
        The size of the memory (number of past picks to store).
    Del : int
        Number of balls to add or replace each step.
    debug : bool
        If True, prints debug info for each iteration.

    Returns:
    -------
    ret_string : list of int
        The sequence of picks (each 0 or 1).
    """
    # Track counts of each color in the urn
    count_R = R
    count_G = G
    count_B = B
    
    # Memory of the last M picks (store color labels here, e.g. 'R', 'G', 'B')
    memory = [None] * M
    
    # The output bit sequence
    ret_string = []
    
    for i in range(len_string):
        # Total number of balls in the urn
        total = count_R + count_G + count_B
        
        # Pick a random index in the combined "urn"
        pick_idx = randrange(total)
        
        # Decide which color was picked based on pick_idx
        if pick_idx < count_B:
            pick_ball = 'B'
            ret_string.append(0)     # B -> 0
        elif pick_idx < count_B + count_R:
            pick_ball = 'R'
            ret_string.append(1)     # R -> 1
        else:
            pick_ball = 'G'
            ret_string.append(1)     # G -> 1 (no 2 in output)
        
        # Remove the oldest from memory, add the newest
        discard_ball = memory.pop(0)
        memory.append(pick_ball)
        
        # Update the urn based on Polya logic
        if discard_ball is None:
            # If discarding None, just add Del copies of the picked color
            if pick_ball == 'B':
                count_B += Del
            elif pick_ball == 'R':
                count_R += Del
            else:  # pick_ball == 'G'
                count_G += Del
        elif discard_ball != pick_ball:
            # We replace up to Del copies of discard_ball with pick_ball
            if discard_ball == 'B':
                replaced = min(count_B, Del)
                count_B -= replaced
                if pick_ball == 'R':
                    count_R += replaced
                else:  # pick_ball == 'G'
                    count_G += replaced
                
            elif discard_ball == 'R':
                replaced = min(count_R, Del)
                count_R -= replaced
                if pick_ball == 'B':
                    count_B += replaced
                else:  # pick_ball == 'G'
                    count_G += replaced
                
            else:  # discard_ball == 'G'
                replaced = min(count_G, Del)
                count_G -= replaced
                if pick_ball == 'B':
                    count_B += replaced
                else:  # pick_ball == 'R'
                    count_R += replaced
        
        if debug:
            print(f"Iteration {i}")
            print(f"  Picked color:  {pick_ball}")
            print(f"  Discard color: {discard_ball}")
            print(f"  Urn counts -> B:{count_B}, R:{count_R}, G:{count_G}")
            print(f"  Memory:     {memory}")
            print(f"  ret_string: {ret_string}")
            print("---------------")
    
    return ret_string

def implement_channel(image_string, R, G, B, M, Del):
#run the channel on a given string
  channel_string = simulate_channel_string(len(image_string), R, G, B, M, Del)
  ret_string = [None] * len(channel_string)
  for i in range(len(channel_string)):
    if (channel_string[i] == 2):
      ret_string[i] = 2
    else:
      ret_string[i] = modulo_add(image_string[i], channel_string[i])
  return ret_string


def channel_with_memory(num_level, epsilon, delta):
    Pr = np.zeros((num_level, num_level))
    n = int(np.log2(num_level))

    # Transition probability matrix for the binary symmetric channel with memory
    Pr_z = np.array([
        [(1 - epsilon + delta) / (1 + delta), epsilon / (1 + delta)],
        [(1 - epsilon) / (1 + delta), (epsilon + delta) / (1 + delta)]
    ])

    for x in range(num_level):
        for y in range(num_level):
            binary_x = np.array([int(bit) for bit in np.binary_repr(x, width=n)])
            binary_y = np.array([int(bit) for bit in np.binary_repr(y, width=n)])
            binary_z = binary_x ^ binary_y  # XOR operation

            if binary_z[0] == 1:
                probability = epsilon
            else:
                probability = 1 - epsilon
            for i in range(1, n):
                probability *= Pr_z[binary_z[i - 1], binary_z[i]]

            Pr[x, y] = probability

    return Pr

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


def encode_image_blocks(img, codebook, b_opt, epsilon, delta, block_size):
    # 1) Convert image blocks into a single 2D array of shape (NB, d)
    blocks = blockify_image(img, block_size)  # list of length NB, each block shape=(d,)
    blocks_array = np.vstack(blocks)          # shape => (NB, d)
    NB = blocks_array.shape[0]
    K  = codebook.shape[0]

    # 2) Build channel probability matrix => shape (K, K)
    P_matrix = channel_with_memory(K, epsilon, delta)

    # WeightedMatrix[i, :] = P_matrix[b_opt[i], :]
    WeightedMatrix = np.zeros((K, K))
    for i_idx in range(K):
        WeightedMatrix[i_idx, :] = P_matrix[b_opt[i_idx], :]

    # 3) Compute Distances: DistMatrix[b, j] = ||blocks_array[b] - codebook[j]||^2
    #    shape => (NB, K)
    Diff = blocks_array[:, np.newaxis, :] - codebook[np.newaxis, :, :]  # (NB, K, d)
    DistMatrix = np.sum(Diff**2, axis=2)                                # (NB, K)

    # 4) WeightedDist = DistMatrix @ WeightedMatrix^T => shape (NB, K)
    #    WeightedDist[b, i] = sum_j P(Y=j|X=b_opt[i]) * DistMatrix[b, j]
    WeightedDist = DistMatrix @ WeightedMatrix.T

    # 5) For each block, pick codeword i minimizing WeightedDist
    encoded_indices = np.argmin(WeightedDist, axis=1)  # shape => (NB,)

    return encoded_indices



def decode_image_blocks(
    received_indices,
    codebook,
    block_size,
    original_shape
):
    H, W = original_shape
    K, d = codebook.shape
    H_blocks = (H + block_size - 1) // block_size
    W_blocks = (W + block_size - 1) // block_size
    NB = len(received_indices)

    # 1) Pre-reshape codewords => shape (K, block_size, block_size) so we skip repeated reshaping
    Codebook3D = codebook.reshape(K, block_size, block_size)

    # 2) Precompute block positions
    row_starts = []
    col_starts = []
    for rb in range(H_blocks):
        for cb in range(W_blocks):
            row_starts.append(rb * block_size)
            col_starts.append(cb * block_size)

    # 3) Fill in the reconstructed image
    reconstructed = np.zeros((H, W), dtype=np.float32)

    for idx, code_idx in enumerate(received_indices):
        r0 = row_starts[idx]
        c0 = col_starts[idx]
        r1 = min(r0 + block_size, H)
        c1 = min(c0 + block_size, W)

        # Directly copy from the 3D codeword block
        block_2d = Codebook3D[code_idx]

        # In case the image was padded, clamp the block
        block_h = r1 - r0
        block_w = c1 - c0

        reconstructed[r0:r1, c0:c1] = block_2d[:block_h, :block_w]

    return reconstructed


def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed)**2)
    if mse < 1e-12: 
        return 100
    # max_val = np.max(original)  # or 255 if image is 8-bit
    return 10 * np.log10(255**2 / mse)




def image_to_bitstring(image_path):
    # Open the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Convert array to bytes
    img_bytes = img_array.tobytes()
    
    # Convert bytes to bit string
    bit_string = ''.join(format(byte, '08b') for byte in img_bytes)

