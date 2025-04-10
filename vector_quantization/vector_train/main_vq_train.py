from sklearn.cluster import KMeans
from vector_quantization.vector_utils.image_utils import *
from pathlib import Path

vector_root = Path(__file__).resolve().parent.parent
project_root = vector_root.parent

image_path = str(project_root / 'images/misc_images')
imgs = [image_path + '/Lena.jpg', image_path + '/Baboon.jpg', image_path + '/satellite0.jpg', image_path + '/satellite1.jpg', image_path + '/satellite2.jpg', image_path + '/satellite3.jpg', image_path + '/satellite4.jpg', image_path + '/satellite5.jpg', image_path + '/satellite6.jpg']

codebooks_path = str(vector_root / '/vector_codebooks/vq')


data = []
block_size = 2

for img in imgs:
  grey_img = grey_image_array(img)
  block_vectors = blockify_image(grey_img,block_size)
  data = data + block_vectors

training_data = (data - np.mean(data)) / np.std(data)

rates = [2,3,4,5,6,7,8]

codebooks = []

for rate in rates:
    num_centroids = pow(2,rate)

    # Train k-means to find the centroids
    kmeans = KMeans(n_clusters=num_centroids, random_state=42)
    kmeans.fit(training_data)

    # The centroids are stored in kmeans.cluster_centers_
    centroids = kmeans.cluster_centers_

    codebooks.append(centroids)
    
np.savez(codebooks_path + "codebooks_" +str(block_size**2) + "dimension.npz", 
             rate2=codebooks[0], rate3=codebooks[1], rate4=codebooks[2], rate5=codebooks[3], rate6=codebooks[4], rate7=codebooks[5], rate8=codebooks[6], mean=np.mean(data), std=np.std(data))