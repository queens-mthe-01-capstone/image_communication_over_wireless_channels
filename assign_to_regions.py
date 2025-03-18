import numpy as np

def assign_to_regions(source_pdf, codebook):
    # Create an empty list to store the Voronoi regions
    Voronoi_regions = [[] for _ in range(len(codebook))]

    for point in source_pdf:
        # Calculate distances from the point to each codebook centroid
        distances = [np.abs(point - c) for c in codebook]
        # Assign the point to the closest centroid (Voronoi region)
        closest_centroid_index = np.argmin(distances)
        Voronoi_regions[closest_centroid_index].append(point)

    return Voronoi_regions