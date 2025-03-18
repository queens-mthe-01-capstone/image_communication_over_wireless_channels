def binary_to_decimal(binary_list):
    decimal = 0
    for bit in binary_list:
        decimal = decimal * 2 + bit
    return decimal