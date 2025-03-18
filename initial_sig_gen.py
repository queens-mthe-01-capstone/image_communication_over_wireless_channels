import numpy as np

# Generate initial signal (splitting algorithm)
def initial_sig_gen(distribution, num_centroids):
    # For Laplace
    if distribution.lower() == 'laplace':
        samples = np.random.laplace(loc=0, scale=np.sqrt(1 / 2), size=500000)
    # For Gaussian
    else:
        samples = np.random.normal(loc=0, scale=1, size=500000)
    min_samples = np.min(samples)
    max_samples = np.max(samples)
    width = (max_samples - min_samples) / num_centroids
    radius = width / 2
    centroids = []
    prob_points = []
    for i in range(num_centroids):
        # Generate centroids
        centroid_current = min_samples + (i + 0.5) * width
        centroids.append(centroid_current)
    for i in range(num_centroids):
        # Experimentally determine probabilities of landing in a certain centroid
        distances = np.abs(samples - centroids[i])
        count = np.sum(distances <= radius)
        prob_points.append(count / len(samples))
    return centroids, radius, prob_points