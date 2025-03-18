'''
def conditional_prob_1(b, i, j, epsilon, delta):
    """
    Compute P(Y=j|X=i) for binary symmetric channel with memory.

    Args:
        b (np.ndarray): Codeword matrix (n_codewords × n_bits)
        i (int): Transmitted codeword index (X=i)
        j (int): Received codeword index (Y=j)
        epsilon (float): Base error probability
        delta (float): Memory effect strength

    Returns:
        float: Probability P(Y=j|X=i)
    """
    n_bits = b.shape[1]
    prob = 1.0

    # First bit (memoryless)
    if modulo_add(b[i, 0], b[j, 0]) == 1:
        prob *= epsilon  # Error in first bit
    else:
        prob *= (1 - epsilon)  # Correct first bit

    # Subsequent bits (Markov-dependent)
    for k in range(1, n_bits):
        # Check if previous bit had error
        prev_error = modulo_add(b[i, k-1], b[j, k-1])

        # Compute error probability for current bit
        error_prob = (epsilon + delta * prev_error) / (1 + delta)

        # Check if current bit has error
        if modulo_add(b[i, k], b[j, k]) == 1:
            prob *= error_prob  # Error in current bit
        else:
            prob *= (1 - error_prob)  # Correct current bit

    return prob
'''
def conditional_prob_1(b, i, j, epsilon, delta, M=1):
    """
    Compute P(Y=j|X=i) for binary symmetric channel with memory based on the given formula.

    Args:
        b (np.ndarray): Codeword matrix (n_codewords × n_bits)
        i (int): Transmitted codeword index (X=i)
        j (int): Received codeword index (Y=j)
        epsilon (float): Base error probability (BER)
        delta (float): Memory effect strength
        M (int): Memory length (number of previous bits affecting the current bit)

    Returns:
        float: Probability P(Y=j|X=i)
    """
    n_bits = b.shape[1]
    prob = 1.0

    # Iterate over each bit
    for k in range(n_bits):
        # Determine the range of previous bits affecting the current bit
        start = max(0, k - M)
        end = k - 1

        # Sum of previous errors within the memory window
        sum_errors = 0
        for m in range(start, end + 1):
            sum_errors += b[i, m] ^ b[j, m]  # XOR operation for modulo-2 addition

        # Compute error probability for the current bit
        error_prob = (epsilon + sum_errors * delta) / (1 + M * delta)

        # Ensure error probability is within [0, 1]
        error_prob = max(0, min(1, error_prob))

        # Check if current bit has error
        current_error = b[i, k] ^ b[j, k]
        if current_error:
            prob *= error_prob  # Error in current bit
        else:
            prob *= (1 - error_prob)  # Correct current bit

    return prob