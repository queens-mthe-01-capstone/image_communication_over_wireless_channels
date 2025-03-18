import matplotlib.pyplot as plt
import numpy as np

def plot_quantization(source_pdf, Voronoi_regions, codebook, iteration=None):

    plt.figure(figsize=(8, 6))

    # Plot data points, with color corresponding to the Voronoi region they belong to
    for i, region in enumerate(Voronoi_regions):
        region_points = np.array(region)
        plt.scatter(region_points, np.zeros_like(region_points), s=10, label=f"Region {i+1}", alpha=0.7)

    # Plot centroids (codebook values)
    codebook_points = np.array(codebook)
    plt.scatter(codebook_points, np.zeros_like(codebook_points), c='red', s=100, marker='X', label="Centroids")

    # Annotate centroids (optional)
    for i, code in enumerate(codebook):
        plt.text(code, 0.02, f'C{i+1}', ha='center', color='red')

    # Label the axes and title
    plt.title(f'Quantization (Iteration {iteration})' if iteration is not None else 'Quantization')
    plt.xlabel('Value')



    # Show the plot
    plt.show()