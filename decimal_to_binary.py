def decimal_to_binary(decimal, num_bits=None):
    if decimal == 0:
        return [0] * (num_bits if num_bits is not None else 1)

    binary = []
    while decimal > 0:
        binary.append(decimal % 2)
        decimal //= 2

    # Reverse the list to get the correct order
    binary.reverse()

    # Pad the binary list with leading zeros if num_bits is specified
    if num_bits is not None:
        binary = [0] * (num_bits - len(binary)) + binary

    return binary