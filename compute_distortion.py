import numpy as np

def compute_distortion(Voronoi_regions, codebook):
    distortion = 0
    for i, region in enumerate(Voronoi_regions):
        centroid = codebook[i]
        # Calculate the squared distance between each point in the region and its centroid
        distortion += np.sum((np.array(region) - centroid) ** 2)
    # Return the average distortion
    return distortion / sum(len(region) for region in Voronoi_regions)  # Normalize by the number of points