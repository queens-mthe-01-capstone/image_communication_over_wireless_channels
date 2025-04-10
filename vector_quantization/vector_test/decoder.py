import numpy as np

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