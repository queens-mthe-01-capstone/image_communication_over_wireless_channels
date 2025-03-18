import numpy as np

def zigzag_order(block):
    """
    Traverse an 8x8 block in zigzag order and return a flattened array.
    """
    rows, cols = block.shape
    result = []
    for diagonal in range(rows + cols - 1):
        if diagonal < cols:
            row = 0
            col = diagonal
        else:
            row = diagonal - cols + 1
            col = cols - 1

        while row < rows and col >= 0:
            result.append(block[row, col])
            row += 1
            col -= 1

    return np.array(result)