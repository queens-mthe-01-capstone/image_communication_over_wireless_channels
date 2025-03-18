from modulo_add import modulo_add

# Conditional probability part of the equation
def conditional_prob(b, j, i, epsilon, delta):
    if modulo_add(b[j, 0], b[i, 0]) == 1:
        p_z1 = epsilon
    else:
        p_z1 = 1 - epsilon
    product_probs = 1
    for k in range(1, b.shape[1]):
        if modulo_add(b[j, k], b[i, k]) == 1:
            product_probs = product_probs * ((epsilon + delta * (modulo_add(b[j, k-1], b[i, k-1]))) / (1 + delta))
        else:
            product_probs = product_probs * (1 - ((epsilon + delta * (modulo_add(b[j, k-1], b[i, k-1]))) / (1 + delta)))
    return p_z1 * product_probs