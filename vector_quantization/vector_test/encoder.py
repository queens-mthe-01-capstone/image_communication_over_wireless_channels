from vector_quantization.vector_utils.image_utils import *
from vector_quantization.vector_utils.quant_utils import *

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
