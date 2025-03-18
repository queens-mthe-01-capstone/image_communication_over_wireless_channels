from conditional_prob_1 import conditional_prob_1
from binary_to_decimal import binary_to_decimal
import numpy as np

def cosq_design(distribution, current_codebook, b_obtained, epsilon, delta=10, tol=1e-4, max_iter=100):

    # Generate source signal (1M samples)
    if distribution.lower() == 'laplace':
        source = np.random.laplace(loc=0, scale=np.sqrt(1/2), size=500000)
    else:  # Gaussian
        source = np.random.normal(loc=0, scale=1, size=500000)

    # Normalize (zero-mean, unit variance)
    source = (source - np.mean(source)) / np.std(source)

    # --------------------------------------------------
    # Critical Fix: Construct P_Y_given_X with TRUE INDICES
    # --------------------------------------------------
    n_codewords = b_obtained.shape[0]
    P_Y_given_X = np.zeros((n_codewords, n_codewords))
    for i in range(n_codewords):
        for j in range(n_codewords):
            P_Y_given_X[i, j] = conditional_prob_1(b_obtained, i, j, epsilon,delta)
            # P_Y_given_X[i, j] = conditional_prob(b_obtained, j, i, epsilon)

    # Initialize codebook and MSE
    # Initialize codebook (ENSURE IT'S A NUMPY ARRAY)
    codebook = np.asarray(current_codebook.copy())  # Convert to NumPy array
    signal_power = np.mean(source ** 2)

    for iteration in range(max_iter):
        # --------------------------------------------------
        # Generalized NNC (Nearest Neighbor Condition)
        # --------------------------------------------------
        # Vectorized distortion calculation (replaces slow loops)
        expanded_source = source[:, np.newaxis]  # Shape (1e6, 1)
        expanded_codebook = codebook[np.newaxis, :]  # Shape (1, n_codewords)

        # Compute (v - c_y)^2 for all v and codewords
        squared_dists = (expanded_source - expanded_codebook) ** 2  # Shape (1e6, n_codewords)

        # Expected distortion: sum_j P(j|i) * (v - c_j)^2
        expected_dists = np.dot(squared_dists, P_Y_given_X.T)  # Shape (1e6, n_codewords)

        # Assign to partition with minimal expected distortion
        partitions = [[] for _ in range(n_codewords)]
        partition_indices = np.argmin(expected_dists, axis=1)
        for idx, p_idx in enumerate(partition_indices):
            partitions[p_idx].append(source[idx])

        # --------------------------------------------------
        # Generalized CC (Centroid Condition)
        # --------------------------------------------------
        new_codebook = np.zeros_like(codebook)
        for y_idx in range(n_codewords):
            numerator = 0.0
            denominator = 0.0

            for i in range(n_codewords):
                if len(partitions[i]) == 0:
                    continue

                # Critical Fix: Use TRUE INDEX i (not permuted)
                prob = P_Y_given_X[i, y_idx]
                sum_v = np.sum(partitions[i])
                count = len(partitions[i])

                numerator += prob * sum_v
                denominator += prob * count

            if denominator > 1e-10:
                new_codebook[y_idx] = numerator / denominator
            else:
                new_codebook[y_idx] = codebook[y_idx]  # No update

        # Check convergence
        codebook_change = np.max(np.abs(new_codebook - codebook))
        codebook = new_codebook.copy()
        if codebook_change < tol:
            break

    mse_q = 0
    # For partition i
    for i in range(n_codewords):
        for x in partitions[i]:
            mse_q = mse_q + (x - codebook[i])**2
    mse_q = mse_q /len(source)

    mse_c = 0
    for i in range(n_codewords):
        j = binary_to_decimal(b_obtained[i])
        for k in range(n_codewords):
            prod = P_Y_given_X[j,k]*(codebook[j]-codebook[k])**2
            mse_c += prod
    mse_c = mse_c / n_codewords
    # Final SNR calculation
    # Or maybe SDR?
    noise_distortion = mse_q + mse_c
    print(f"Noise distortion: {noise_distortion}")
    snr = 10 * np.log10(signal_power / noise_distortion)

    return codebook, partitions, snr, source