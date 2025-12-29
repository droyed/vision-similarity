# Vision Similarity

A Python library for computing visual similarity between images using state-of-the-art vision transformer models from Hugging Face.

## Overview

Vision Similarity provides a simple API for extracting deep learning features from images and computing similarity scores. Built on PyTorch and Hugging Face Transformers, it supports multiple pretrained models (DINOv2, CLIP, etc.) and enables efficient batch processing with GPU acceleration.

**Common use cases:**
- Finding duplicate or near-duplicate images
- Image retrieval and search systems
- Visual similarity analysis and clustering
- Dataset deduplication

## Key Features

- **Multiple pretrained models**: DINOv2 (small/base/large/giant), CLIP, and other vision transformers
- **Flexible comparison modes**: Compare two image sets or perform self-similarity analysis
- **Feature extraction strategies**: Mean patches, CLS token, or mean of all tokens
- **Efficient processing**: Batch processing with GPU support and memory management
- **Rich visualizations**: Generate similarity heatmaps with image thumbnails
- **Persistent feature stores**: Save and reuse extracted features in NPZ format
- **Modern Python packaging**: Built with pyproject.toml following PEP 517/518/621

## Installation

### Quick Install

```bash
# Install from GitHub repository
pip install git+https://github.com/yourusername/vision-similarity.git

# Install from GitHub with development tools
pip install "git+https://github.com/yourusername/vision-similarity.git#egg=vision-similarity[dev]"

# Local installation (if you've cloned the repo)
pip install -e .

# Local installation with development tools
pip install -e ".[dev]"
```

### GPU Support (Recommended)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Requirements:** Python 3.8+, pip 19.0+

For detailed installation instructions, troubleshooting, and environment setup, see [docs/installation.md](docs/installation.md).

## Quick Start

### Compare Two Image Sets

```python
from vision_similarity import FeatureSimilarity

# Create similarity analyzer
similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-base",
    image_paths_set1=["cat1.jpg", "cat2.jpg"],
    image_paths_set2=["dog1.jpg", "dog2.jpg"],
    batch_size=5,
    metric="cosine"
)

# Compute similarity matrix
sim_matrix = similarity.compute()
print(sim_matrix)

# Generate heatmap visualization
similarity.plot(
    savepath="similarity_heatmap.png",
    title="Cats vs Dogs Similarity"
)
```

### Self-Similarity Analysis

```python
# Compare images within the same set
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-base",
    image_paths_set1=image_paths,
    image_paths_set2=None  # None triggers self-comparison
)

sim_matrix = similarity.matrix
```

## Usage Overview

### Working with Feature Stores

Extract features once and reuse them for multiple comparisons:

```python
from vision_similarity import ImageFeatureStore, cosine_similarity_matrix

# Extract and save features
store = ImageFeatureStore.from_images(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    pretrained_model_name="facebook/dinov2-base",
    strategy="mean_patches"
)

# Save for later use
store.save_npz("features.npz")

# Load and compute similarities
loaded_store = ImageFeatureStore.from_npz("features.npz")
sim = cosine_similarity_matrix(loaded_store.features, loaded_store.features)
```

### List Available Models

```python
from vision_similarity import print_models_table

# Print all models without verification (faster)
print_models_table(verify_availability=False)

# Or verify availability on Hugging Face Hub
print_models_table(verify_availability=True)

# Customize table format
print_models_table(tablefmt='fancy_grid')
```

### Process Image Directories

```python
from vision_similarity import list_image_files, FeatureSimilarity

# Automatically find all images in directories
cats = list_image_files("path/to/cats/", sort=True)
dogs = list_image_files("path/to/dogs/", sort=True)

similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-large",
    image_paths_set1=cats,
    image_paths_set2=dogs,
    batch_size=16
)

similarity.plot(savepath="cats_vs_dogs.png")
```

### Feature Extraction Strategies

- **`mean_patches`** (recommended): Average of spatial patch tokens - best for similarity
- **`cls`**: Use CLS token - faster but may be less accurate
- **`mean_all`**: Average of all tokens including CLS

### GPU vs CPU

```python
from vision_similarity import load_vision_model

# Auto-detect GPU
model, processor = load_vision_model(
    pretrained_model_name="facebook/dinov2-base",
    device_map="auto"
)

# Force CPU
model, processor = load_vision_model(
    pretrained_model_name="facebook/dinov2-base",
    device_map="cpu"
)
```

## Documentation

- **[installation.md](docs/installation.md)** - Detailed installation guide, dependencies, and troubleshooting
- **[quick_start.md](docs/quick_start.md)** - Project structure, common commands, and quick reference
- **[usage_examples.md](docs/usage_examples.md)** - 10+ detailed examples covering all features

Additional documentation:
- `TESTING.md` - How to run tests
- `PACKAGING.md` - Package structure and build process
- `DEPENDENCIES.md` - Dependency management
- `SETUP_VS_PYPROJECT.md` - Modern packaging approach

## Important Notes

### Model Dependencies

Some models may require Hugging Face authentication:

```bash
export HUGGINGFACE_TOKEN="your_token_here"
# or
huggingface-cli login
```

### Performance Considerations

- **Batch size**: Adjust based on GPU memory (4-32 typical range)
- **Model size**: Larger models (dinov2-large, dinov2-giant) provide better features but require more memory
- **Memory cleanup**: Use `cleanup=True` in batch processing to free GPU memory between batches

### Reproducibility

```python
from vision_similarity import make_deterministic

make_deterministic(seed=42)
```

## Running Tests

```bash
# Using test runner
./run_tests.sh
# or
python3 run_tests.py

# Individual tests
python3 tests/test_vision_similarity.py
```

## Package Structure

```
vision-similarity/
├── src/vision_similarity/      # Main package
│   ├── __init__.py
│   ├── vision_similarity.py
│   ├── utils.py
│   ├── list_models.py
│   ├── plot_similarity_matrix.py
│   └── configs/models.yaml
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml              # Package configuration
└── requirements.txt            # Pinned dependencies
```

## Attributions

Images used in this project are from the Oxford-IIIT Pet Dataset by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar, available at https://www.robots.ox.ac.uk/~vgg/data/pets/ under the Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/by-sa/4.0/). Copyright remains with the original image owners.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Divakar Roy
