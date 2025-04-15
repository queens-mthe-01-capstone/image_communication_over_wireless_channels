
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
    """
    Returns a num_level x num_level matrix Pr[x,y] = P(Y=y|X=x),
    taking into account the memory factor in the bit transitions.
    """
    Pr = np.zeros((num_level, num_level))
    n = int(np.log2(num_level))

    # Transition probabilities for each bit (conditional)
    Pr_z = np.array([
        [(1 - epsilon + delta) / (1 + delta), epsilon / (1 + delta)],
        [(1 - epsilon) / (1 + delta), (epsilon + delta) / (1 + delta)]
    ])

    for x in range(num_level):
        for y in range(num_level):
            binary_x = np.array([int(bit) for bit in np.binary_repr(x, width=n)])
            binary_y = np.array([int(bit) for bit in np.binary_repr(y, width=n)])
            binary_z = binary_x ^ binary_y  # XOR operation

            # Probability for first bit
            probability = epsilon if binary_z[0] == 1 else 1 - epsilon
            # Probability for subsequent bits
            for i in range(1, n):
                probability *= Pr_z[binary_z[i - 1], binary_z[i]]

            Pr[x, y] = probability

    return Pr

# Parallelized simulated annealing
def simulated_annealing(T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, epsilon, delta, centroids, partitions, num_centroids):
    """
    Performs simulated annealing to find a good permutation `b`.
    """
    T = T_0
    count = 0
    count_success = 0
    count_fail = 0
    b_history = []

    # Probability of each partition (based on how many samples each partition holds)
    # The original code assumes 500000 samples total.
    prob_points = np.array([len(partition) / 500000 for partition in partitions])
    conditional_prob = channel_with_memory(num_centroids, epsilon, delta)

    # Precompute squares for centroids to avoid repeated operations
    centroids_sq = centroids**2

    while T > T_f and count_fail < N_fail:
        b_prime = random.sample(b, len(b))

        # Vectorized distortion for b
        # shape => (num_centroids, num_centroids) for conditional_prob[b,...]
        # We'll do a matrix approach:

        # 1) difference[i, j] = (centroids[b[i]] - centroids[j])^2
        # Then multiply by prob_points[i] * conditional_prob[b[i], j]
        # Then sum i,j.

        # We can build difference matrix:
        cb = centroids[b]      # shape (num_centroids,)
        cb_sq = cb**2
        # difference[i,j] = cb[i]^2 - 2 * cb[i]*centroids[j] + centroids[j]^2
        # We'll do this in a vectorized manner:
        # shape (num_centroids, num_centroids)
        diff_b = (
            cb_sq[:, None]
            - 2.0 * np.outer(cb, centroids)
            + centroids_sq[None, :]
        )

        # Weighted by prob_points[i] * conditional_prob[b[i], j]
        # We'll construct pmat[i, j] = prob_points[i] * conditional_prob[b[i], j]
        pmat_b = prob_points[:, None] * conditional_prob[b, :]
        distortion_b = np.sum(pmat_b * diff_b)

        # Vectorized distortion for b_prime
        cbp = centroids[b_prime]
        cbp_sq = cbp**2
        diff_bprime = (
            cbp_sq[:, None]
            - 2.0 * np.outer(cbp, centroids)
            + centroids_sq[None, :]
        )
        pmat_bprime = prob_points[:, None] * conditional_prob[b_prime, :]
        distortion_b_prime = np.sum(pmat_bprime * diff_bprime)

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

def cosq_design(source, current_codebook, epsilon, b_obtained, num_centroids, tol=1e-4, max_iter=100):
    """
    Vectorized COSQ design for a given codebook, noise level epsilon, 
    and permutation b_obtained. Returns updated codebook, partitions, and SNR.
    """
    n_codewords = len(current_codebook)
    # Cache channel matrix for faster reuse
    P_Y_given_X = channel_with_memory(n_codewords, epsilon, 10)

    codebook = np.asarray(current_codebook).copy()
    signal_power = np.mean(source**2)
    
    # Pre-allocate memory for partitions
    partition_indices = np.zeros_like(source, dtype=np.int32)

    for iteration in range(max_iter):
        # -------- 1) Vectorized Distortion Computation --------
        # We want distortions[i, s] = sum_{j}( P_Y_given_X[j, b_obtained[i]] * (source[s] - codebook[j])^2 ).
        # Instead of nested loops, do matrix multiplication:
        #
        #   diffs[j, s] = (source[s] - codebook[j])^2
        #   p[i, j]     = P_Y_given_X[j, b_obtained[i]]
        #
        #   distortions[i, :] = p[i, :] @ diffs[:, :] = sum_j [ p[i,j] * diffs[j,s] ]
        #
        # Build 'diffs' once:
        diffs = (source[None, :] - codebook[:, None])**2  # shape (n_codewords, #samples)

        # Build p[i, j] for i in [0..n_codewords-1]
        # p[i, :] = P_Y_given_X[:, b_obtained[i]]
        p_matrix = np.array([P_Y_given_X[:, b_obtained[i]] for i in range(n_codewords)])  # shape (n_codewords, n_codewords)

        # distortions => shape (n_codewords, #samples)
        distortions = p_matrix @ diffs  # (n_codewords, n_codewords) @ (n_codewords, #samples) => (n_codewords, #samples)

        # -------- 2) Partition Assignment --------
        # Choose the partition index i that yields min distortion for each sample
        partition_indices = np.argmin(distortions, axis=0)  # shape (#samples,)

        # -------- 3) Recompute Centroids --------
        # We'll use sums, sums_of_squares, and lengths of each partition.
        # However, here the channel weighting is in the codebook update:
        #
        #   codebook[j] = [ sum_{i} ( P_Y_given_X[i, b_obtained[j]] * sum(partitions[i]) ) ]
        #                / [ sum_{i} ( P_Y_given_X[i, b_obtained[j]] * len(partitions[i]) ) ]
        #
        # Instead of physically slicing partitions, we use np.bincount:
        sums_part = np.bincount(partition_indices, weights=source, minlength=n_codewords)
        counts_part = np.bincount(partition_indices, minlength=n_codewords)

        new_codebook = codebook.copy()

        # We'll do a loop over j in [0..n_codewords-1], but inside use dot products for speed.
        for j in range(n_codewords):
            # weights for each partition i => P_Y_given_X[i, b_obtained[j]]
            w_ij = P_Y_given_X[np.arange(n_codewords), b_obtained[j]]
            
            # Weighted sum of partition i means: sum_{i} [w_ij[i] * sums_part[i]]
            numerator = w_ij.dot(sums_part)
            denominator = w_ij.dot(counts_part)
            if denominator > 0:
                new_codebook[j] = numerator / denominator
            else:
                new_codebook[j] = codebook[j]  # no change if no samples in denominator

        # Check for convergence
        if np.max(np.abs(new_codebook - codebook)) < tol:
            codebook = new_codebook
            break

        codebook = new_codebook

    # -------- 4) Final Partition & MSE Calculation --------
    # One more pass: recompute partition_indices with final codebook
    diffs = (source[None, :] - codebook[:, None])**2
    p_matrix = np.array([P_Y_given_X[:, b_obtained[i]] for i in range(n_codewords)])
    distortions = p_matrix @ diffs
    partition_indices = np.argmin(distortions, axis=0)

    # Instead of storing large arrays for each partition, we can store them if needed.
    # We'll build them as a list of arrays (or just store indices).
    partitions = []
    for i in range(n_codewords):
        idx_i = np.where(partition_indices == i)[0]
        partitions.append(source[idx_i])

    # Compute final MSE using sums-of-squares trick:
    # error = sum_{i,j} [ P_Y_given_X[b_obtained[i], j] * sum_{x in partition i}(x - codebook[j])^2 ]
    # We'll use precomputed sums, sums-of-squares, etc.:
    sums_part = np.bincount(partition_indices, weights=source, minlength=n_codewords)  # sum of x in partition i
    sumsqr_part = np.bincount(partition_indices, weights=source*source, minlength=n_codewords)  # sum of x^2 in partition i
    counts_part = np.bincount(partition_indices, minlength=n_codewords)

    codebook_sq = codebook**2
    total_error = 0.0
    for i in range(n_codewords):
        A = sumsqr_part[i]       # \sum x^2 in partition i
        B = sums_part[i]         # \sum x    in partition i
        L = counts_part[i]       # #samples in partition i
        for j in range(n_codewords):
            c = codebook[j]
            # sum_{x in partition i} (x - c)^2 = A - 2cB + c^2 * L
            cost_ij = A - 2.0*c*B + c*c*L
            total_error += P_Y_given_X[b_obtained[i], j] * cost_ij

    mse = total_error / len(source)
    snr = 10.0 * np.log10(signal_power / mse)

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

        codebooks_set = []
        partitions_set = []

        # Initialize b
        init_b = list(range(num_centroids))

        # 1) Generate initial codebook
        init_codebook = generate_initial_codebook(sampled_source, num_centroids)
        codebooks_set.append(np.array(init_codebook))
        partitions_set.append([])  # No partition yet for the initial guess

        # 2) COSQ for noiseless
        noiseless_codebook, noiseless_partition, noiseless_snr = cosq_design(
            sampled_source, codebooks_set[-1], 
            epsilon=1e-11, 
            b_obtained=init_b, 
            num_centroids=num_centroids
        )
        codebooks_set.append(np.array(noiseless_codebook))
        partitions_set.append([p.tolist() for p in noiseless_partition])

        # 3) Simulated Annealing on the noiseless partitions
        b_from_sa = simulated_annealing(
            T_0, alpha, T_f, init_b, 
            N_fail, N_success, N_cut, k, 
            0.005, delta, 
            noiseless_codebook, noiseless_partition, 
            num_centroids=num_centroids
        )

        # 4) Generate codebooks for different noise levels
        epsilons = [0.005, 0.01, 0.05, 0.1, 0.05, 0.01, 0.005]
        for epsilon in tqdm(epsilons, desc=f"Processing epsilons for rate {rate}", leave=False):
            codebook, partition, snr = cosq_design(
                sampled_source, codebooks_set[-1], 
                epsilon, b_from_sa, 
                num_centroids=num_centroids
            )
            codebooks_set.append(np.array(codebook))
            partitions_set.append([p.tolist() for p in partition])

        # 5) Save the codebooks and partitions for the current rate
        for idx, codebook in enumerate(codebooks_set):
            # Create keys for the codebook and the partitions
            key_codebook = f"{codebook_type}_rate_{rate}_set_{idx}_codebook"

            codebooks_dict[key_codebook] = codebook.tolist()


    # Finally, save all codebooks and partitions to the JSON file
    with open(output_filename, 'w') as f:
        json.dump(codebooks_dict, f, indent=4)
