from generate_initial_codebook import generate_initial_codebook
from cosq_design import cosq_design
import numpy as np

def train_cosq_with_noise_levels(initial_distribution, num_centroids, noise_levels, T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, delta, distribution='laplace'):

    # Start by generating the initial codebook with no noise
    initial_b, initial_codebook = generate_initial_codebook(T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, 0, delta, distribution, num_centroids)

    index_assign = []
    codebooks = []

    index_assign.append(np.array(initial_b))
    # codebooks.append(np.array(no_noise_codebook))
    codebooks.append(np.array(initial_codebook))

    noiseless_codebook, noiseless_partition, snr, source_pdf = cosq_design(initial_distribution, codebooks[-1], b, 1e-11, 1e-6, delta)


    print("Noiseless SNR:",snr)
    codebooks.append(np.array(noiseless_codebook))




    # Now proceed with the remaining noise levels, using the codebook from the previous step
    for epsilon in noise_levels:

        print(f"Training with noise level: {epsilon}")

      # print("b before cosq",type(b_obtained))

       # Update the codebook using the previous noise level's codebook
        # b_obtained, current_codebook = generateInitialCodebook(T_0, alpha, T_f, b, N_fail, N_success, N_cut, k, epsilon, delta, distribution, num_centroids)



        final_codebook, final_partition, snr, source_pdf = cosq_design(initial_distribution, codebooks[-1], initial_b, epsilon, 1e-6)

        codebooks.append(np.array(final_codebook))

        # Print the results for each noise level
        print(f"Final Codebook at noise level {epsilon}:", final_codebook)
        print(f"SNR at noise level {epsilon}:", snr)

    return codebooks