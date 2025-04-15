import json
import os
import sys
import numpy as np
import math
import random
from tqdm import tqdm
from functools import lru_cache
from joblib import Parallel, delayed

# Vectorized binary array generation
def generate_binary_array(num_centroids):
    k = int(np.log2(num_centroids))
    return np.array([list(map(int, np.binary_repr(i, width=k))) for i in range(num_centroids)])

# Memoize the channel_with_memory function to avoid redundant computations
@lru_cache(maxsize=None)
def channel_with_memory(num_level, epsilon, delta):
    Pr = np.zeros((num_level, num_level))
    n = int(np.log2(num_level))

    Pr_z = np.array([
        [(1 - epsilon + delta) / (1 + delta), epsilon / (1 + delta)],
        [(1 - epsilon) / (1 + delta), (epsilon + delta) / (1 + delta)]
    ])

    for x in range(num_level):
        for y in range(num_level):
            binary_x = np.array([int(bit) for bit in np.binary_repr(x, width=n)])
            binary_y = np.array([int(bit) for bit in np.binary_repr(y, width=n)])
            binary_z = binary_x ^ binary_y  # XOR operation

            probability = epsilon if binary_z[0] == 1 else 1 - epsilon
            for i in range(1, n):
                probability *= Pr_z[binary_z[i - 1], binary_z[i]]

            Pr[x, y] = probability

    return Pr

# Parallelized simulated annealing
def simulated_annealing(T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, epsilon, delta, centroids, partitions, num_centroids):
    T = T_0
    count = 0
    count_success = 0
    count_fail = 0
    b_history = []

    prob_points = np.array([len(partition) / 500000 for partition in partitions])
    conditional_prob = channel_with_memory(num_centroids, epsilon, delta)

    while T > T_f and count_fail < N_fail:
        b_prime = random.sample(b, len(b))

        # Vectorized distortion calculation
        distortion_b = np.sum(prob_points[:, None] * conditional_prob[b] * ((centroids[b] - centroids[:, None]) ** 2))
        distortion_b_prime = np.sum(prob_points[:, None] * conditional_prob[b_prime] * ((centroids[b_prime] - centroids[:, None]) ** 2))

        delta_Dc = (distortion_b_prime - distortion_b) / k
        b_history.append(distortion_b)

        if delta_Dc <= 0 or random.uniform(0, 1) <= math.exp(-delta_Dc / T):
            b = b_prime
            count_success += 1
            count_fail = 0
        else:
            count_fail += 1

        if count >= N_cut or count_success >= N_success:
            T *= alpha
            count = 0
            count_success = 0
        count += 1

    return b

# Vectorized source signal generation
def generate_source_signal(distribution, num_samples=500000):
    if distribution.lower() == 'laplace':
        source = np.random.laplace(loc=0, scale=np.sqrt(1 / 2), size=num_samples)
    else:
        source = np.random.normal(loc=0, scale=1, size=num_samples)

    return (source - np.mean(source)) / np.std(source)

# Vectorized initial codebook generation
def generate_initial_codebook(source, num_centroids):
    min_samples = np.min(source)
    max_samples = np.max(source)
    width = (max_samples - min_samples) / num_centroids
    return [min_samples + (i + 0.5) * width for i in range(num_centroids)]

# Parallelized COSQ design
def cosq_design(source, current_codebook, epsilon, b_obtained, num_centroids, tol=1e-4, max_iter=100):
    """
    Returns:
        codebook (np.ndarray): Updated codebook array.
        partitions (list of np.ndarray): Each element is an array of the samples
                                         that were assigned to that codeword.
        snr (float): The resulting SNR after the final iteration.
    """
    n_codewords = len(current_codebook)
    P_Y_given_X = channel_with_memory(n_codewords, epsilon, 10)

    codebook = np.asarray(current_codebook.copy())
    signal_power = np.mean(source ** 2)

    for iteration in range(max_iter):
        # Compute distortions for each sample and centroid
        distortions = np.zeros((num_centroids, len(source)))  
        for i in range(num_centroids):
            for j in range(num_centroids):
                # Contribution of being assigned to centroid j if X=i, weighted by channel
                distortions[i] += P_Y_given_X[j, b_obtained[i]] * ((source - codebook[j]) ** 2)

        # Assign each sample to the partition with the minimum distortion
        partition_indices = np.argmin(distortions, axis=0)
        partitions = [source[partition_indices == i] for i in range(num_centroids)]

        # Update centroids
        new_codebook = np.zeros_like(codebook)
        for j in range(n_codewords):
            numerator = 0.0
            denominator = 0.0
            for i in range(num_centroids):
                if len(partitions[i]) > 0:
                    numerator += P_Y_given_X[i, b_obtained[j]] * np.sum(partitions[i])
                    denominator += P_Y_given_X[i, b_obtained[j]] * len(partitions[i])
            # Avoid division by zero
            if denominator > 0:
                new_codebook[j] = numerator / denominator
            else:
                new_codebook[j] = codebook[j]

        # Check for convergence
        if np.max(np.abs(new_codebook - codebook)) < tol:
            codebook = new_codebook
            break
        codebook = new_codebook

    # Calculate MSE and SNR
    total_error = 0.0
    for i in range(num_centroids):
        for j in range(num_centroids):
            total_error += P_Y_given_X[b_obtained[i], j] * np.sum((partitions[i] - codebook[j]) ** 2)
    mse = total_error / len(source)
    snr = 10 * np.log10(signal_power / mse)

    return codebook, partitions, snr


def generate_and_save_codebooks(rates, codebook_type):
    # Parameters for SA
    T_0 = 10
    alpha = 0.97
    T_f = 0.00025
    N_fail = 50000
    N_success = 5
    N_cut = 200
    k = 10
    # For channel
    delta = 10

    if codebook_type.lower() == "ac":
        initial_distribution = "laplace"
    else:
        initial_distribution = "gaussian"

    # Load existing codebooks if the file exists
    output_filename = 'codebooks.json'
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            codebooks_dict = json.load(f)
    else:
        codebooks_dict = {}

    # Process each rate
    for rate in tqdm(rates, desc=f"Processing {codebook_type} codebooks"):
        num_centroids = 2 ** rate
        sampled_source = generate_source_signal(initial_distribution)
        
        # We'll keep track of each (codebook, partition) set
        codebooks_set = []
        partitions_set = []

        # Initialize b
        init_b = list(range(num_centroids))
        
        # 1) Generate initial codebook
        init_codebook = generate_initial_codebook(sampled_source, num_centroids)
        codebooks_set.append(np.array(init_codebook))
        # Partitions for the initial codebook are not well-defined (it hasn't gone through a COSQ iteration),
        # but if you wish, you could do a quick partition assignment here. We'll store an empty placeholder:
        partitions_set.append([])  # or do a simple assignment if you like

        # 2) COSQ for noiseless
        noiseless_codebook, noiseless_partition, noiseless_snr = cosq_design(
            sampled_source, codebooks_set[-1], epsilon=1e-11, 
            b_obtained=init_b, num_centroids=num_centroids
        )
        codebooks_set.append(np.array(noiseless_codebook))
        partitions_set.append([p.tolist() for p in noiseless_partition])  # store partitions as lists

        # 3) Simulated Annealing on the noiseless partitions
        b_from_sa = simulated_annealing(
            T_0, alpha, T_f, init_b, 
            N_fail, N_success, N_cut, 
            k, 0.005, delta, 
            noiseless_codebook, noiseless_partition, 
            num_centroids=num_centroids
        )

        # 4) Generate codebooks for different noise levels
        epsilons = [0.005, 0.01, 0.05, 0.1, 0.05, 0.01, 0.005]
        for epsilon in tqdm(epsilons, desc=f"Processing epsilons for rate {rate}", leave=False):
            codebook, partition, snr = cosq_design(
                sampled_source, codebooks_set[-1], epsilon, 
                b_from_sa, num_centroids=num_centroids
            )
            codebooks_set.append(np.array(codebook))
            partitions_set.append([p.tolist() for p in partition])

        # 5) Save the codebooks and partitions for the current rate
        for idx, codebook in enumerate(codebooks_set):
            # Create two keys: one for the codebook, one for the partitions
            key_codebook = f"{codebook_type}_rate_{rate}_set_{idx}_codebook"
            key_partitions = f"{codebook_type}_rate_{rate}_set_{idx}_partitions"
            
            codebooks_dict[key_codebook] = codebook.tolist()
            # In case the idx in partitions_set is empty or doesn't exist, handle carefully:
            if idx < len(partitions_set) and partitions_set[idx]:
                codebooks_dict[key_partitions] = partitions_set[idx]
            else:
                codebooks_dict[key_partitions] = []

    # Finally, save all codebooks and partitions to the JSON file
    with open(output_filename, 'w') as f:
        json.dump(codebooks_dict, f, indent=4)

