"""
OVERVIEW:
---------
This script demonstrates the visualization of image similarity matrices using heatmaps.
It performs the following steps:

1. Image Loading:
   - Scans the 'assets' directory for image files using list_image_files()
   - Loads all discovered images and converts them to RGB format

2. Similarity Matrix Generation:
   - Creates a synthetic similarity matrix with random values between 0.5 and 1.0
   - Ensures the matrix is symmetric (sim[i,j] == sim[j,i])
   - Sets diagonal values to 1.0 to represent perfect self-similarity
   - Matrix dimensions: NxN where N is the number of images found

3. Visualization:
   - Uses plot_matrix() to create a heatmap visualization
   - The heatmap displays:
     * Thumbnail images along both axes
     * Color-coded similarity values (darker = more similar)
     * Custom title for the plot
   - Saves the resulting visualization to 'heatmap.png'

INPUTS:
-------
- Image directory: 'assets' (configurable)
- Random seed: 42 (for reproducibility)

OUTPUTS:
--------
- Console: Prints the similarity matrix values and save location
- File: 'heatmap.png' - Visual heatmap with image thumbnails and similarity scores

USE CASE:
---------
This demo is useful for understanding how to visualize pairwise similarity
relationships between images, such as those computed by feature extraction
models (e.g., SAM embeddings, CLIP features, etc.)
"""

import numpy as np
from PIL import Image
from vision_similarity.plot_similarity_matrix import plot_matrix
from vision_similarity.utils import list_image_files

# Image paths
imgpaths = list_image_files('assets', sort=True)

# Load images
images = [Image.open(path).convert('RGB') for path in imgpaths]

# Create random symmetric similarity matrix (3x3)
np.random.seed(42)
num_images = len(images)    
sim_matrix = np.random.rand(num_images, num_images) * 0.5 + 0.5  # Values between 0.5 and 1.0
sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
np.fill_diagonal(sim_matrix, 1.0)  # Diagonal = 1.0 (self-similarity)

print(f"Similarity matrix:\n{sim_matrix}\n")

savepath = 'heatmap.png'

# Plot similarity heatmap
canvas = plot_matrix(
    images1=images,
    images2=images,
    sim=sim_matrix,
    savepath=savepath,
    title="Image Similarity Heatmap"
)

print(f"Heatmap saved to: {savepath}")
