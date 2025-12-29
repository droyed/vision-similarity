"""
Vision Similarity.

A Python package for computing and visualizing image similarity
using pre-trained vision models.
"""

from .vision_similarity import (
    ImageFeatureStore,
    FeatureSimilarity,
    load_vision_model,
    extract_image_features,
    cosine_similarity_matrix,
    make_deterministic,
)

from .list_models import (
    get_available_models as list_models,
    get_all_available_models_list as list_models_flat,
    print_models_table,
)

from .plot_similarity_matrix import (
    plot_matrix as plot_similarity_matrix,
)

from .utils import (
    list_image_files,
)

__version__ = "0.1.0"

__all__ = [
    # Core workflow
    "ImageFeatureStore",
    "FeatureSimilarity",
    "load_vision_model",
    "extract_image_features",
    "cosine_similarity_matrix",

    # Discovery
    "list_models",
    "list_models_flat",
    "print_models_table",

    # Utilities
    "list_image_files",
    "make_deterministic",

    # Visualization
    "plot_similarity_matrix",
]
