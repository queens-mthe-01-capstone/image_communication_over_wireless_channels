import numpy as np
import json
import os
from tqdm import tqdm

def generate_source_signal(distribution: str, num_samples=500000) -> np.ndarray:
    """
    Generate 'num_samples' from either a Laplacian or Gaussian distribution.
    Normalize to zero-mean, unit-variance.
    """
    if distribution.lower() == 'laplace':
        source = np.random.laplace(loc=0, scale=np.sqrt(1 / 2), size=num_samples)
    else:
        source = np.random.normal(loc=0, scale=1, size=num_samples)
    # Normalize
    source = (source - np.mean(source)) / np.std(source)
    return source

def lloyd_max_quantizer_1d(data: np.ndarray, num_centroids: int, 
                           max_iter=100, tol=1e-5) -> np.ndarray:
    """
    Standard 1D Lloyd–Max quantizer:
      1) Initialize codebook (centroids) uniformly across data range.
      2) Iteratively refine:
         - compute partition boundaries = midpoints between consecutive centroids
         - assign points to nearest centroid
         - update centroid = mean of assigned points
    Returns the final codebook (sorted).
    """
    # 1) Initialize codebook uniformly across the data range
    dmin, dmax = data.min(), data.max()
    codebook = np.linspace(dmin, dmax, num_centroids)

    for _ in range(max_iter):
        old_codebook = codebook.copy()

        # 2) Build partition boundaries (midpoints)
        boundaries = 0.5 * (codebook[:-1] + codebook[1:])  # length = num_centroids - 1

        # 3) Assign each sample to the nearest centroid
        # We'll do a fast approach: use np.searchsorted for boundaries
        # Then compare with last centroid if needed
        idx = np.searchsorted(boundaries, data)  # gives intervals in [0..num_centroids-1]
        
        # 4) Update each centroid as the mean of assigned points
        for i in range(num_centroids):
            pts = data[idx == i]
            if len(pts) > 0:
                codebook[i] = pts.mean()

        # 5) Check for convergence
        if np.max(np.abs(codebook - old_codebook)) < tol:
            break

    # Sort codebook (useful if the data distribution is unusual, 
    # though it should typically remain sorted).
    codebook.sort()
    return codebook

def generate_and_save_codebooks(rates, distribution='laplace', output_filename='lloydmax_codebooks.json'):
    """
    For each 'rate' in 'rates', produce a Lloyd–Max codebook of size 2^rate,
    store them in 'codebooks.json'.
    """
    # Load existing codebooks if the file exists
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            codebooks_dict = json.load(f)
    else:
        codebooks_dict = {}

    # Generate source data
    print(f"Generating source data from '{distribution}' distribution...")
    data = generate_source_signal(distribution)

    # For each rate, build the Lloyd–Max codebook
    for r in tqdm(rates, desc="Generating Lloyd–Max codebooks"):
        num_centroids = 2 ** r

        print(f"Building Lloyd–Max quantizer: rate={r}, num_centroids={num_centroids}")
        codebook = lloyd_max_quantizer_1d(data, num_centroids)

        # Store in dictionary
        key_codebook = f"{distribution}_rate_{r}_lloydmax_codebook"
        codebooks_dict[key_codebook] = codebook.tolist()

    # Save to JSON
    with open(output_filename, 'w') as f:
        json.dump(codebooks_dict, f, indent=4)
