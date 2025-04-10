import numpy as np
import random
import math
from random import randrange

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

# Simulated Annealing Algorithm
def simulated_annealing(T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, epsilon, delta, centroids, partitions, num_centroids, num_source_samples):
    T = T_0
    count = 0
    count_success = 0
    count_fail = 0
    b_history = []

    prob_points = []

    # Loop over each partition
    for partition in partitions:
        # Calculate the probability of samples falling in this partition
        prob = len(partition) / num_source_samples
        prob_points.append(prob)

    conditional_prob = channel_with_memory(num_centroids, epsilon, delta)

    while T > T_f and count_fail < N_fail:

        b_prime = random.sample(b, len(b))

        delta_Dc = 0

        distortion_b = 0
        distortion_b_prime = 0

        for g in range(0, num_centroids):
            for h in range(0, num_centroids):
                distortion_b = distortion_b + prob_points[g] * conditional_prob[h,b[g]] * np.sum((centroids[b[g]]-centroids[h])**2)
                distortion_b_prime = distortion_b_prime + prob_points[g] * conditional_prob[h, b_prime[g]] * np.sum((centroids[b_prime[g]]-centroids[h])**2)

        distortion_b = distortion_b * (1 / k)
        distortion_b_prime = distortion_b_prime * (1 / k)
        delta_Dc = distortion_b_prime - distortion_b
        b_history.append(distortion_b)

        if delta_Dc <= 0:
            b = b_prime
            count_success = count_success + 1
            count_fail = 0
        else:
            rand_num = random.uniform(0, 1)
            if rand_num <= math.exp(-delta_Dc / T):
                b = b_prime
            count_fail = count_fail + 1

        if count >= N_cut or count_success >= N_success:
            T = alpha * T
            count = 0
            count_success = 0
        count = count + 1

    return b

def generate_initial_codebook(source, num_centroids):

    min_vals = np.min(source, axis=0)
    max_vals = np.max(source, axis=0)

    centroids = []
    for i in range(num_centroids):
        alpha = (i + 0.5) / num_centroids
        centroid = min_vals + alpha * (max_vals - min_vals)
        centroids.append(centroid)

    return np.array(centroids)

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


def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed)**2)
    if mse < 1e-12: 
        return 100
    # max_val = np.max(original)  # or 255 if image is 8-bit
    return 10 * np.log10(255**2 / mse)
