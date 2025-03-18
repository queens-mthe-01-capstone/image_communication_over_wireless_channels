import numpy as np

def number_to_binary_numpy(number, length):
    """
    Convert a number to a binary representation of a fixed length using NumPy.
    """
    if number < 0 or number >= 2**length:
        raise ValueError(f"Number {number} cannot be represented with {length} bits.")

    # Create a NumPy array with the number
    arr = np.array([number], dtype=np.uint8)

    # Unpack the number into binary bits
    binary_array = np.unpackbits(arr)[-length:]

    return binary_array