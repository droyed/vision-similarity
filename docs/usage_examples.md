# Usage Examples

This document provides practical examples for using the `vision-similarity` package.

## Quick Start

### 1. Basic Image Similarity Comparison

```python
from vision_similarity import FeatureSimilarity

# Compare two sets of images
image_paths_set1 = ["path/to/image1.jpg", "path/to/image2.jpg"]
image_paths_set2 = ["path/to/image3.jpg", "path/to/image4.jpg"]

# Create similarity analyzer
similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-base",
    image_paths_set1=image_paths_set1,
    image_paths_set2=image_paths_set2,
    batch_size=5,
    metric="cosine",
    strategy="mean_patches"
)

# Compute similarity matrix
sim_matrix = similarity.compute()
print(f"Similarity matrix shape: {sim_matrix.shape}")
print(sim_matrix)

# Visualize with heatmap
similarity.plot(
    savepath="similarity_heatmap.png",
    title="Image Similarity Matrix",
    cell_px=128,
    cmap_name="YlGnBu"
)
```

### 2. Compare Images Within a Single Set (Self-Similarity)

```python
from vision_similarity import FeatureSimilarity

# Compare images within the same set
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]

similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-small",
    image_paths_set1=image_paths,
    image_paths_set2=None,  # None means self-comparison
    batch_size=4
)

sim_matrix = similarity.matrix
print(f"Self-similarity matrix:\n{sim_matrix}")
```

### 3. Using Image Feature Stores

```python
from vision_similarity import ImageFeatureStore, cosine_similarity_matrix

# Extract and save features for later use
store1 = ImageFeatureStore.from_images(
    image_paths=["cat1.jpg", "cat2.jpg", "cat3.jpg"],
    pretrained_model_name="facebook/dinov2-base",
    batch_size=8,
    strategy="mean_patches"
)

# Save features to disk
store1.save_npz("cat_features.npz")

# Load features later
store1_loaded = ImageFeatureStore.from_npz("cat_features.npz")

# Extract features for another set
store2 = ImageFeatureStore.from_images(
    image_paths=["dog1.jpg", "dog2.jpg", "dog3.jpg"],
    pretrained_model_name="facebook/dinov2-base",
    batch_size=8
)

# Compute similarity between the two stores
sim = cosine_similarity_matrix(store1_loaded.features, store2.features)
print(f"Cross-category similarity:\n{sim}")
```

### 4. List Available Models

```python
from vision_similarity import print_models_table

# Print all available models with their information
print_models_table(verify_availability=False, tablefmt='grid')

# Or verify availability on Hugging Face Hub
print_models_table(verify_availability=True, tablefmt='fancy_grid')

# Get just the model names (if needed for programmatic access)
from vision_similarity import get_all_available_models_list
model_names = get_all_available_models_list(verify_availability=False)
print(f"Available models: {model_names}")
```

### 5. Working with Directory of Images

```python
from vision_similarity import list_image_files, FeatureSimilarity

# List all images in a directory
images_set1 = list_image_files("path/to/cats/", sort=True)
images_set2 = list_image_files("path/to/dogs/", sort=True)

print(f"Found {len(images_set1)} cat images")
print(f"Found {len(images_set2)} dog images")

# Compare all cats vs all dogs
similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-large",
    image_paths_set1=images_set1,
    image_paths_set2=images_set2,
    batch_size=16
)

# Generate heatmap
similarity.plot(
    savepath="cats_vs_dogs_similarity.png",
    title="Cats vs Dogs Similarity",
    cell_px=100,
    thumb_pad=4,
    gridline_px=2,
    cmap_name="RdYlGn"
)
```

### 6. Using Different Feature Extraction Strategies

```python
from vision_similarity import ImageFeatureStore

image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

# Strategy 1: Mean of patch tokens (recommended)
store_patches = ImageFeatureStore.from_images(
    image_paths=image_paths,
    pretrained_model_name="facebook/dinov2-base",
    strategy="mean_patches"
)

# Strategy 2: CLS token
store_cls = ImageFeatureStore.from_images(
    image_paths=image_paths,
    pretrained_model_name="facebook/dinov2-base",
    strategy="cls"
)

# Strategy 3: Mean of all tokens
store_all = ImageFeatureStore.from_images(
    image_paths=image_paths,
    pretrained_model_name="facebook/dinov2-base",
    strategy="mean_all"
)

print(f"Patches strategy features: {store_patches.features.shape}")
print(f"CLS strategy features: {store_cls.features.shape}")
print(f"All tokens strategy features: {store_all.features.shape}")
```

### 7. GPU vs CPU Usage

```python
from vision_similarity import load_vision_model, ImageFeatureStore

# Load model on GPU (if available)
model_gpu, processor = load_vision_model(
    pretrained_model_name="facebook/dinov2-base",
    device_map="auto"  # Automatically uses GPU if available
)

# Or explicitly specify device
model_cpu, processor_cpu = load_vision_model(
    pretrained_model_name="facebook/dinov2-base",
    device_map="cpu"  # Force CPU usage
)

# Use the loaded model
store = ImageFeatureStore.from_images_with_model(
    image_paths=["img1.jpg", "img2.jpg"],
    model=model_gpu,
    processor=processor,
    batch_size=8
)
```

### 8. Deterministic Results

```python
from vision_similarity import make_deterministic, ImageFeatureStore

# Set seed for reproducible results
make_deterministic(seed=42)

# Now feature extraction will be deterministic
store = ImageFeatureStore.from_images(
    image_paths=["img1.jpg", "img2.jpg"],
    pretrained_model_name="facebook/dinov2-base"
)
```

### 9. Hugging Face Authentication

```python
from vision_similarity import login_huggingface_from_env

# Login using environment variable
success = login_huggingface_from_env(
    env_var="HUGGINGFACE_TOKEN",
    verbose=True
)

if success:
    print("Successfully logged in to Hugging Face!")
else:
    print("Login failed or token not found")
```

### 10. Custom Visualization Settings

```python
from vision_similarity import plot_matrix
from PIL import Image
import numpy as np

# Load images
images1 = [Image.open(f"cat{i}.jpg") for i in range(1, 4)]
images2 = [Image.open(f"dog{i}.jpg") for i in range(1, 4)]

# Create dummy similarity matrix for demonstration
sim_matrix = np.random.rand(3, 3)

# Custom visualization
canvas = plot_matrix(
    images1=images1,
    images2=images2,
    sim=sim_matrix,
    cell_px=150,           # Larger cells
    thumb_pad=6,           # More padding
    gridline_px=3,         # Thicker gridlines
    cmap_name="viridis",   # Different colormap
    vmin=0.0,
    vmax=1.0,
    savepath="custom_viz.png",
    title="Custom Similarity Visualization",
    block=False
)

print(f"Canvas shape: {canvas.shape}")
```

## Common Use Cases

### Use Case 1: Finding Duplicate or Similar Images

```python
from vision_similarity import ImageFeatureStore, cosine_similarity_matrix
import numpy as np

# Extract features from all images in a directory
from vision_similarity import list_image_files
all_images = list_image_files("my_images/", sort=True)

store = ImageFeatureStore.from_images(
    image_paths=all_images,
    pretrained_model_name="facebook/dinov2-base"
)

# Compute self-similarity
sim = cosine_similarity_matrix(store.features, store.features)

# Find near-duplicates (similarity > 0.95, excluding self)
threshold = 0.95
duplicates = []
for i in range(len(all_images)):
    for j in range(i + 1, len(all_images)):
        if sim[i, j] > threshold:
            duplicates.append((all_images[i], all_images[j], sim[i, j]))

print(f"Found {len(duplicates)} potential duplicate pairs:")
for img1, img2, score in duplicates:
    print(f"  {img1} <-> {img2} (similarity: {score:.3f})")
```

### Use Case 2: Image Retrieval/Search

```python
from vision_similarity import ImageFeatureStore, cosine_similarity_matrix
import numpy as np

# Build a database of reference images
reference_images = list_image_files("image_database/")
db_store = ImageFeatureStore.from_images(
    image_paths=reference_images,
    pretrained_model_name="facebook/dinov2-base"
)
db_store.save_npz("image_db_features.npz")

# Query with a new image
query_store = ImageFeatureStore.from_images(
    image_paths=["query_image.jpg"],
    pretrained_model_name="facebook/dinov2-base"
)

# Find most similar images
sim = cosine_similarity_matrix(query_store.features, db_store.features)
top_k = 5
top_indices = np.argsort(sim[0])[-top_k:][::-1]

print(f"Top {top_k} most similar images:")
for idx in top_indices:
    print(f"  {reference_images[idx]} (similarity: {sim[0, idx]:.3f})")
```

## Advanced Configuration

### Batch Size Optimization

```python
# Small batch size for limited GPU memory
store = ImageFeatureStore.from_images(
    image_paths=images,
    pretrained_model_name="facebook/dinov2-large",
    batch_size=4  # Reduce if getting OOM errors
)

# Large batch size for better performance
store = ImageFeatureStore.from_images(
    image_paths=images,
    pretrained_model_name="facebook/dinov2-small",
    batch_size=32  # Increase for faster processing
)
```

### Memory Cleanup

```python
from vision_similarity import extract_image_features, load_vision_model

model, processor = load_vision_model("facebook/dinov2-base")

# Enable aggressive memory cleanup
features = extract_image_features(
    image_paths=images,
    model=model,
    processor=processor,
    batch_size=8,
    cleanup=True  # Frees GPU memory after each batch
)
```

## Tips and Best Practices

1. **Choose the right model**: Larger models (dinov2-large, dinov2-giant) provide better features but are slower
2. **Use appropriate batch sizes**: Adjust based on your GPU memory
3. **Save feature stores**: Extract features once and reuse them
4. **Use mean_patches strategy**: Generally provides best similarity results
5. **Normalize your images**: The models handle this automatically via the processor
6. **Set deterministic mode**: For reproducible experiments

For more details, refer to the API documentation and source code.
