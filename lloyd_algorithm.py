from assign_to_regions import assign_to_regions
from compute_distortion import compute_distortion
from update_codebook import update_codebook
from plot_quantization import plot_quantization
import numpy as np

# Lloyd max algorithm for noiseless
def lloyd_algorithm(distribution, initial_codebook):

    # Generate the source signal
    if distribution.lower() == 'laplace':
        source_pdf = np.random.laplace(loc=0, scale=np.sqrt(1/2), size=500000) #Use 500k

    else:  # Gaussian
        source_pdf = np.random.normal(loc=0, scale=1, size=500000)


    # Initialize codebook and Voronoi regions
    codebook = initial_codebook.copy()
    Voronoi_regions = []  # Initialize an empty list for Voronoi regions
    distortion_history = []

    while True:
        # Step 1: Assign each point in the PDF to the closest codebook entry (Voronoi regions)
        Voronoi_regions = assign_to_regions(source_pdf, codebook)  # Make sure Voronoi_regions is updated here

        # Step 2: Update codebook by computing the centroids of each Voronoi region
        new_codebook = update_codebook(Voronoi_regions)

        # Step 3: Compute distortion (sum of squared distances between points and their closest codebook entry)
        distortion = compute_distortion(Voronoi_regions, new_codebook)

        distortion_history.append(distortion)


        # Step 4: Check for convergence (if distortion change is below threshold)
        if len(distortion_history) > 1 and abs(distortion_history[-1] - distortion_history[-2]) < 1e6:
            break

        # Step 5: Update codebook for the next iteration
        codebook = new_codebook

    signal_power = np.mean(source_pdf**2)
    noise_power = distortion_history[-1]
    snr = 10 * np.log10(signal_power / noise_power)
    plot_quantization(source_pdf, Voronoi_regions, codebook)
    print(snr)
    return Voronoi_regions, codebook, source_pdf, snr  # Return final Voronoi regions, codebook, and distortion