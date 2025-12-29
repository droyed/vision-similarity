# Installation Guide

This guide provides instructions for installing and setting up the `vision-similarity` package.

## Prerequisites

- Python 3.8 or higher
- pip 19.0+ (for modern packaging support)
- (Optional) CUDA-enabled GPU for faster processing

## Important: Modern Packaging

This package uses **modern Python packaging (PEP 517/518/621)** with `pyproject.toml`.
**No setup.py is needed** - everything is configured in `pyproject.toml`.

## Installation Methods

### Method 1: Development Installation (Recommended for Contributors)

For development or if you want to modify the code:

```bash
# Clone or navigate to the project directory
cd /path/to/vision-similarity

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e .

# Or install with development tools (includes pytest, black, flake8, etc.)
pip install -e ".[dev]"
```

### Method 2: Installation from Source

```bash
# Navigate to the project directory
cd /path/to/vision-similarity

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install using pip (reads from pyproject.toml)
pip install .
```

### Method 3: Install from Built Distribution

```bash
# First, build the package
pip install build
python -m build

# Then install the wheel
pip install dist/vision_similarity-0.1.0-py3-none-any.whl
```

## Installing Dependencies

### Core Dependencies

The package requires the following core dependencies:

- `Pillow`: Image processing
- `numpy`: Numerical computing
- `matplotlib`: Visualization
- `torch`: PyTorch deep learning framework
- `torchvision`: Computer vision tools
- `transformers`: Hugging Face transformers library
- `huggingface-hub`: Hugging Face Hub API
- `PyYAML`: YAML parsing
- `pandas`: Data manipulation
- `tabulate`: Table formatting

### GPU Support

For GPU acceleration (recommended):

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only installation:

```bash
pip install torch torchvision
```

### Development Dependencies

To install development dependencies for testing and linting:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest
- pytest-cov
- black
- flake8
- isort
- mypy

## Hugging Face Authentication

Some models may require authentication with Hugging Face Hub:

```bash
# Set your Hugging Face token as an environment variable
export HUGGINGFACE_TOKEN="your_token_here"

# Or use the huggingface-cli
pip install huggingface-hub
huggingface-cli login
```

## Verification

After installation, verify that the package is installed correctly:

```python
# Test import
python -c "from vision_similarity import __version__; print(f'Version: {__version__}')"

# Test basic functionality
python -c "from vision_similarity import get_available_models; models = get_available_models(verify_availability=False); print(f'Available models: {len(sum(models.values(), []))}')"
```

## Package Structure

After installation, the package structure is:

```
vision-similarity/
├── src/
│   └── vision_similarity/
│       ├── __init__.py
│       ├── vision_similarity.py
│       ├── utils.py
│       ├── list_models.py
│       └── plot_similarity_matrix.py
├── configs/
│   └── models.yaml
├── tests/
├── setup.py
├── pyproject.toml
├── requirements.txt
└── MANIFEST.in
```

## Building Distribution Packages

To build distribution packages (wheel and source):

```bash
# Install build tools
pip install build

# Build the package
python -m build

# This creates:
# - dist/vision_similarity-0.1.0-py3-none-any.whl
# - dist/vision_similarity-0.1.0.tar.gz
```

## Uninstallation

To uninstall the package:

```bash
pip uninstall vision-similarity
```

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure you've activated your virtual environment and installed all dependencies:

```bash
source venv/bin/activate
pip install -e .
```

### Issue: CUDA/GPU errors

**Solution**: Install the correct PyTorch version for your CUDA setup:

```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Permission errors during installation

**Solution**: Use a virtual environment instead of system-wide installation:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue: Git dependency installation fails

**Solution**: Install git and ensure you have internet connectivity:

```bash
# On Ubuntu/Debian
sudo apt-get install git

# Then retry installation
pip install -e .
```

## Next Steps

After installation, refer to the documentation for:

- Usage examples
- API reference
- Configuration options
- Model selection guide

For more information, see the project documentation.
