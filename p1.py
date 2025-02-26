import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import rasterio
from rasterio.enums import Resampling

# Define file paths for Sentinel-2 bands
red_band_path = r"redband.jp2"
green_band_path = r"greenband.jp2"
nir_band_path = r"nirband.jp2"
swir_band_path = r"swir.jp2"

# Load Red, Green, and NIR bands
def load_band(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1), src.profile

red_band, profile = load_band(red_band_path)
green_band, _ = load_band(green_band_path)
nir_band, _ = load_band(nir_band_path)

# Load and Resample SWIR band
def resample_band(file_path, shape):
    with rasterio.open(file_path) as src:
        return src.read(1, out_shape=shape, resampling=Resampling.bilinear)

swir_band = resample_band(swir_band_path, (red_band.shape[0], red_band.shape[1]))

# Stack bands to create a multispectral image
image = np.stack([red_band, green_band, nir_band, swir_band], axis=-1)

# Downsample image to speed up clustering
def downsample(image, factor=4):
    return image[::factor, ::factor, :]

image = downsample(image)

# Convert image to feature space
pixels = image.reshape(-1, image.shape[-1])

# ISODATA Clustering Implementation
def isodata_clustering(data, num_clusters=4, max_iter=10, min_size=10):
    np.random.seed(42)
    cluster_centers = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
    labels = np.zeros(data.shape[0], dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        for i in range(num_clusters):
            cluster_data = data[labels == i]
            if len(cluster_data) > min_size:
                cluster_centers[i] = cluster_data.mean(axis=0)
    
    return labels.reshape(image.shape[:2])

# Run ISODATA clustering
isodata_clustered = isodata_clustering(pixels, num_clusters=4)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(np.stack([red_band, green_band, nir_band], axis=-1) / np.max(image))
ax[0].set_title("Original Sentinel Image (Red, Green, NIR, SWIR)")
ax[1].imshow(isodata_clustered, cmap='jet')
ax[1].set_title("ISODATA Clustering")
for a in ax:
    a.axis("off")
plt.show()