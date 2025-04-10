from vector_utils.quant_utils import *

def covq_design(source, current_codebook, epsilon, b_obtained, delta, tol=1e-4, max_iter=100):
    N = source.shape[0]
    K, d = current_codebook.shape

    # Build channel probability matrix
    P_Y_given_X = channel_with_memory(K, epsilon, delta)

    # WeightedMatrix[i, :] = P_Y_given_X[b_obtained[i], :]
    # -> used to vectorize partition step
    WeightedMatrix = np.zeros((K, K))
    for i_idx in range(K):
        WeightedMatrix[i_idx, :] = P_Y_given_X[b_obtained[i_idx], :]

    codebook = current_codebook.copy()

    signal_power = np.mean(np.sum(source**2, axis=1))

    for iteration in range(max_iter):
        # ----------------------------------------------------------
        # 1) PARTITION STEP 
        # ----------------------------------------------------------
        Diff = source[:, np.newaxis, :] - codebook[np.newaxis, :, :]  
        DistMatrix = np.sum(Diff**2, axis=2)                       

        WeightedDist = DistMatrix @ WeightedMatrix.T

        partition_indices = np.argmin(WeightedDist, axis=1)

        # Build partition lists
        partitions = [[] for _ in range(K)]
        for n, i_idx in enumerate(partition_indices):
            partitions[i_idx].append(source[n])

        # ----------------------------------------------------------
        # 2) CENTROID STEP 
        # ----------------------------------------------------------
        new_codebook = np.zeros_like(codebook)

        # Precompute sums & counts for each partition j
        PartitionSum = np.zeros((K, d), dtype=np.float64)
        PartitionCount = np.zeros(K, dtype=np.float64)

        for j_idx in range(K):
            if len(partitions[j_idx]) > 0:
                block_j = np.vstack(partitions[j_idx])  # shape (#vectors_j, d)
                PartitionSum[j_idx] = np.sum(block_j, axis=0)
                PartitionCount[j_idx] = len(block_j)

        # Now update each codeword i
        for i_idx in range(K):
            numerator = np.zeros(d, dtype=np.float64)
            denominator = 0.0

            for j_idx in range(K):
                if PartitionCount[j_idx] == 0:
                    continue
                # p_ij = P(Y=i_idx | X=b_obtained[j_idx])
                # But your code uses P_Y_given_X[i_idx, b_obtained[j_idx]] 
                # Carefully ensure consistent indexing
                p_ij = P_Y_given_X[i_idx, b_obtained[j_idx]]
                numerator   += p_ij * PartitionSum[j_idx]
                denominator += p_ij * PartitionCount[j_idx]

            # If no weight, keep old codeword
            if denominator < 1e-12:
                new_codebook[i_idx] = codebook[i_idx]
            else:
                new_codebook[i_idx] = numerator / denominator

        # ----------------------------------------------------------
        # 3) Check Convergence
        # ----------------------------------------------------------
        # If the max codeword shift is < tol, we stop
        codebook_change = np.max(np.linalg.norm(new_codebook - codebook, axis=1))
        codebook = new_codebook
        # print(codebook)
        if codebook_change < tol:
            break

    # ----------------------------------------------------------
    # 4) Final Distortion + SNR
    # ----------------------------------------------------------
    # We'll do one last partition assignment to measure MSE precisely
    Diff = source[:, np.newaxis, :] - codebook[np.newaxis, :, :]
    DistMatrix = np.sum(Diff**2, axis=2)  # shape (N, K)
    WeightedDist = DistMatrix @ WeightedMatrix.T
    final_partition_indices = np.argmin(WeightedDist, axis=1)

    # Compute MSE by summation
    # We'll do a second approach:
    #   MSE = average of expected distortion for each sample with final assignment
    # Or replicate your original loop. We'll do it vectorized:
    min_vals = WeightedDist[np.arange(N), final_partition_indices]
    mse = np.mean(min_vals)
    snr = 10 * np.log10(signal_power / mse)

    # If you still want the final partitions, build them again:
    final_partitions = [[] for _ in range(K)]
    for n, i_idx in enumerate(final_partition_indices):
        final_partitions[i_idx].append(source[n])

    return codebook, final_partitions, snr