"""
List all available models for feature extraction.
This script queries Hugging Face Hub for available models from the list of models in models.yaml.
"""

from pathlib import Path
from typing import Dict, List
import yaml
from transformers import AutoImageProcessor
from tabulate import tabulate
import pandas as pd


def _load_models_config() -> Dict[str, List[Dict]]:
    """Load models configuration from YAML file."""
    # Try multiple locations for the config file
    # 1. Inside package (src/vision_similarity/configs/models.yaml)
    # 2. At project root (configs/models.yaml)
    config_locations = [
        Path(__file__).parent / "configs" / "models.yaml",  # Inside package
        Path(__file__).parent.parent.parent / "configs" / "models.yaml",  # Project root
    ]

    for config_path in config_locations:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find models.yaml config file. Tried: {[str(p) for p in config_locations]}"
    )


def get_available_models(verify_availability: bool = True) -> Dict[str, List[str]]:
    """
    Gather all available models for feature extraction.

    Args:
        verify_availability: If True, checks each model's availability on Hugging Face Hub.
                           If False, returns all models without verification (faster).

    Returns:
        Dictionary with model categories as keys and lists of available model names as values.
        Structure dynamically matches the categories in models.yaml.
    """
    # Load models from YAML configuration
    models_config = _load_models_config()

    # Extract model IDs by category (generalized for any categories in config)
    models_by_category = {
        category: [model['id'] for model in models]
        for category, models in models_config.items()
    }

    # If not verifying, return all models
    if not verify_availability:
        return models_by_category

    # Verify availability by trying to load the processor
    def check_model(model_name: str) -> bool:
        """Check if a model is available."""
        try:
            AutoImageProcessor.from_pretrained(model_name)
            return True
        except Exception:
            return False

    # Filter models by availability for each category
    available_models = {
        category: [m for m in models if check_model(m)]
        for category, models in models_by_category.items()
    }

    return available_models


def get_all_available_models_list(verify_availability: bool = True) -> List[str]:
    """
    Get a flat list of all available models.

    Args:
        verify_availability: If True, checks each model's availability on Hugging Face Hub.
                           If False, returns all models without verification (faster).

    Returns:
        List of all available model names.
    """
    models_dict = get_available_models(verify_availability=verify_availability)
    # Flatten all model lists from all categories
    all_models = []
    for models in models_dict.values():
        all_models.extend(models)
    return all_models


def get_available_models_with_info(verify_availability: bool = True) -> Dict[str, List[Dict[str, str]]]:
    """
    Get available models with additional information.

    Args:
        verify_availability: If True, checks each model's availability on Hugging Face Hub.
                           If False, returns all models without verification (faster).

    Returns:
        Dictionary with model categories as keys and lists of model info dicts as values.
        Each model info dict contains: {'name': model_name, 'params': params, 'size': size_info, 'description': desc, 'hidden_dim': hidden_dim}
    """
    # Load models from YAML configuration
    models_config = _load_models_config()

    # Transform YAML data to match expected output format
    model_info = {}
    for category, models in models_config.items():
        model_info[category] = [
            {
                'name': model['id'],
                'params': model.get('params', 'Unknown'),
                'size': model.get('size_info', model.get('params', 'Unknown')),
                'description': model.get('description', ''),
                'hidden_dim': model.get('hidden_dim', 'N/A')
            }
            for model in models
        ]

    if not verify_availability:
        return model_info

    # Verify availability
    def check_model(model_name: str) -> bool:
        """Check if a model is available."""
        try:
            AutoImageProcessor.from_pretrained(model_name)
            return True
        except Exception:
            return False

    # Filter by availability
    result = {}
    for category, models in model_info.items():
        result[category] = [m for m in models if check_model(m['name'])]

    return result


def print_models_table(verify_availability: bool = True, tablefmt: str = 'fancy_outline'):
    """
    Print models information in a nicely formatted table using pandas multi-index DataFrame.

    Args:
        verify_availability: If True, checks each model's availability on Hugging Face Hub.
                           If False, returns all models without verification (faster).
        tablefmt: Table format style. Options include:
                 'fancy_outline' (default), 'grid', 'fancy_grid', 'simple', 'plain', 'simple_grid',
                 'rounded_grid', 'heavy_grid', 'mixed_grid', 'double_grid',
                 'github', 'pipe', 'orgtbl', 'jira', 'presto',
                 'pretty', 'psql', 'rst', 'mediawiki', 'moinmoin', 'youtrack',
                 'html', 'latex', 'latex_raw', 'latex_booktabs', 'textile'
    """
    # Generate models_info internally
    models_info = get_available_models_with_info(verify_availability=verify_availability)

    # Create multi-index DataFrame
    data_rows = []

    # Category name mapping for better display
    category_names = {
        'dinov3': 'DINOv3 Models (Facebook)',
        'dinov2': 'DINOv2 Models (Facebook)',
        'other': 'Other Vision Models'
    }

    for category, models in models_info.items():
        category_display = category_names.get(category, category.upper())
        for model in models:
            data_rows.append({
                'Category': category_display,
                'Model Name': model['name'],
                'Params': model.get('params', 'Unknown'),
                'Hidden Dim': model.get('hidden_dim', 'N/A'),
                'Size': model['size'],
                'Description': model['description']
            })

    if not data_rows:
        print("No models available.")
        return

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Set multi-index with Category and Model Name
    df_multi = df.set_index(['Category', 'Model Name'])

    # Reset index to show multi-index as columns for better tabulate display
    df_display = df_multi.reset_index()

    # Replace repeated category labels with empty strings for cleaner display
    df_display_clean = df_display.copy()
    prev_category = None
    for idx, row in df_display.iterrows():
        if row['Category'] == prev_category:
            df_display_clean.at[idx, 'Category'] = ''
        else:
            prev_category = row['Category']

    # Print using tabulate for consistent formatting
    print(tabulate(df_display_clean, headers='keys', tablefmt=tablefmt, showindex=False))
