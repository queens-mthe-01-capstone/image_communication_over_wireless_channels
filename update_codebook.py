import numpy as np

def update_codebook(Voronoi_regions):
    # Calculate the new centroids (codebook) as the mean of points in each Voronoi region
    new_codebook = []

    for region in Voronoi_regions:
        if len(region) > 0:
            new_codebook.append(np.mean(region))
        else:
            new_codebook.append(0)  # Default to 0 if the region has no points assigned (though this should not happen)

    return new_codebook