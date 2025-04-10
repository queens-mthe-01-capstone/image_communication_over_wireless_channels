from vector_quantization.vector_utils.image_utils import *
from vector_quantization.vector_utils.quant_utils import *
from vector_quantization.vector_train.covq_design import *
from pathlib import Path

vector_root = Path(__file__).resolve().parent.parent
project_root = vector_root.parent

image_path = str(project_root / 'images/misc_images')
imgs = [image_path + '/Lena.jpg', image_path + '/Baboon.jpg', image_path + '/satellite0.jpg', image_path + '/satellite1.jpg', image_path + '/satellite2.jpg', image_path + '/satellite3.jpg', image_path + '/satellite4.jpg', image_path + '/satellite5.jpg', image_path + '/satellite6.jpg']

codebooks_path = str(vector_root / '/vector_codebooks/covq')

block_size = 2
rates = [2,3,4,5,6,7,8]
epsilons = [0,0.005,0.01,0.05,0.1]

data = []

for img in imgs:
  grey_img = grey_image_array(img)
  block_vectors = blockify_image(grey_img,block_size)
  data = data + block_vectors

training_data = (data - np.mean(data)) / np.std(data)

for rate in rates:
    codebooks = []

    num_centroids = pow(2,rate)
    init_codebook = generate_initial_codebook(training_data, num_centroids)

    init_b = list(range(num_centroids))

    T_0 = 10
    alpha = 0.97
    T_f = 0.00025
    N_fail = 5000
    N_success = 5
    N_cut = 50
    k = 10
    # For channel
    delta = 10

    codebook, partitions, snr = covq_design(
        training_data,
        init_codebook,
        1e-11,  
        init_b,
        delta
    )

    # b_opt = simulated_annealing(T_0, alpha, T_f, init_b, N_fail, N_success, N_cut, k, 0.005, delta, codebook, partitions, num_centroids, len(training_data))

    for epsilon in epsilons:
        codebook, partitions, snr = covq_design(
            training_data,
            codebook,
            epsilon,
            init_b,
            delta
        )

        codebooks.append(codebook)

    np.savez(codebooks_path + "codebooks_" + str(block_size**2) + "dimension_" + str(2**rate) + "codewords.npz", 
             epsilon0=codebooks[0], epsilon1=codebooks[1], epsilon2=codebooks[2], epsilon3=codebooks[3], epsilon4=codebooks[4], mean=np.mean(data), std=np.std(data))

