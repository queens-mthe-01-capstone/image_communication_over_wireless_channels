from initial_sig_gen import initial_sig_gen
from distance import distance
import math
import random
from conditional_prob_1 import conditional_prob_1
import numpy as np

# Simulated Annealing Algorithm
def generate_initial_codebook(T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, epsilon, delta, distribution, num_centroids):
    T = T_0
    count = 0
    count_success = 0
    count_fail = 0
    b_history = []

    # Generate signal and centroids
    centroids, radius, prob_points = initial_sig_gen(distribution, num_centroids)

    while T > T_f and count_fail < N_fail:
        b_prime = b[np.random.permutation(b.shape[0])]
        delta_Dc = 0
        distortion_b = 0
        distortion_b_prime = 0
        for g in range(0, b.shape[0]):
            for h in range(0, b.shape[0]):
                # distortion_b = distortion_b + prob_points[g] * conditional_prob(b, h, g, epsilon) * distance(centroids[g], centroids[h])
                # distortion_b_prime = distortion_b_prime + prob_points[g] * conditional_prob(b_prime, h, g, epsilon) * distance(centroids[g], centroids[h])

                distortion_b = distortion_b + prob_points[g] * conditional_prob_1(b,g, h, epsilon,delta) * distance(centroids[g], centroids[h])
                distortion_b_prime = distortion_b_prime + prob_points[g] * conditional_prob_1(b_prime, g, h, epsilon,delta) * distance(centroids[g], centroids[h])


        distortion_b = distortion_b * (1 / k)
        distortion_b_prime = distortion_b_prime * (1 / k)
        delta_Dc = distortion_b_prime - distortion_b
        b_history.append(distortion_b)
        if delta_Dc <= 0:
            b = b_prime
            count_success = count_success + 1
            count_fail = 0
        else:
            rand_num = random.uniform(0, 1)
            if rand_num <= math.exp(-delta_Dc / T):
                b = b_prime
            count_fail = count_fail + 1
        if count >= N_cut or count_success >= N_success:
            T = alpha * T
            count = 0
            count_success = 0
        count = count + 1

    # plt.plot(b_history)
    # plt.title('Cost Function Over Time')
    # plt.xlabel('Iteration Number')
    # plt.ylabel('Cost Function')
    # plt.show()

    return b, centroids