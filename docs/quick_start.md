# Quick Start Guide

## Running Tests

### Method 1: Use Test Runner (Recommended)

```bash
# Bash (Linux/Mac)
./run_tests.sh

# Python (Cross-platform)
python3 run_tests.py
```

### Method 2: Run Individual Tests

```bash
# Run specific test
python3 tests/test_vision_similarity.py
python3 tests/test_list_models.py
python3 tests/test_plot_matrix.py
```

### Method 3: Run All Tests Manually

```bash
# One by one
python3 tests/test_list_models.py
python3 tests/test_vision_similarity.py
python3 tests/test_plot_matrix.py
python3 tests/test_similarity_workflows.py
python3 tests/test_generate_feature_similarity_heatmaps.py
```

---

## Installation

### Quick Install

```bash
# Standard installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With exact versions (reproducibility)
pip install -r requirements-lock.txt
```

---

## Project Structure

```
vision-similarity/
├── pyproject.toml          # Package configuration
├── requirements-lock.txt   # Pinned dependencies
├── run_tests.sh            # Test runner (bash)
├── run_tests.py            # Test runner (python)
│
├── src/vision_similarity/  # Package source
│   ├── __init__.py
│   ├── configs/models.yaml
│   ├── vision_similarity.py
│   ├── utils.py
│   ├── list_models.py
│   └── plot_similarity_matrix.py
│
├── tests/                  # Test scripts
│   ├── test_vision_similarity.py
│   ├── test_list_models.py
│   ├── test_plot_matrix.py
│   ├── test_similarity_workflows.py
│   └── test_generate_feature_similarity_heatmaps.py
│
└── docs/                   # Documentation
    ├── INSTALLATION.md
    ├── USAGE_EXAMPLES.md
    ├── PACKAGING.md
    ├── TESTING.md
    ├── DEPENDENCIES.md
    └── ...
```

---

## Common Commands

| Task | Command |
|------|---------|
| **Install package** | `pip install -e .` |
| **Run all tests** | `./run_tests.sh` or `python3 run_tests.py` |
| **Run one test** | `python3 tests/test_vision_similarity.py` |
| **Build package** | `python3 -m build` |
| **Format code** | `black src/` |
| **Sort imports** | `isort src/` |
| **Lint code** | `flake8 src/` |
| **Type check** | `mypy src/` |

---

## Documentation Files

| File | Description |
|------|-------------|
| **INSTALLATION.md** | How to install the package |
| **USAGE_EXAMPLES.md** | 10+ code examples |
| **TESTING.md** | How to run tests |
| **PACKAGING.md** | Package structure & build |
| **DEPENDENCIES.md** | Dependency management |
| **SETUP_VS_PYPROJECT.md** | Why no setup.py |
| **REQUIREMENTS_DECISION.md** | About requirements.txt |

---

## Quick Examples

### Example 1: Compare Images

```python
from vision_similarity import FeatureSimilarity

similarity = FeatureSimilarity.from_image_sets(
    pretrained_model_name="facebook/dinov2-base",
    image_paths_set1=["cat1.jpg", "cat2.jpg"],
    image_paths_set2=["dog1.jpg", "dog2.jpg"],
)

sim_matrix = similarity.compute()
similarity.plot(savepath="output.png")
```

### Example 2: List Available Models

```python
from vision_similarity import print_models_table

# Print all models without verification (faster)
print_models_table(verify_availability=False)
```

---

## Troubleshooting

### Import Error
```bash
# Install package
pip install -e .
```

### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements-lock.txt
```

### CUDA Out of Memory
```bash
# Force CPU mode
CUDA_VISIBLE_DEVICES="" python3 tests/test_vision_similarity.py
```

---

## Need Help?

- 📖 Read [INSTALLATION.md](INSTALLATION.md)
- 💡 Check [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
- 🧪 See [TESTING.md](TESTING.md)
- 📦 Review [PACKAGING.md](PACKAGING.md)

---

## Modern Python Packaging

This project uses:
- ✅ **pyproject.toml** (not setup.py)
- ✅ **requirements-lock.txt** (reproducibility)
- ✅ **src/ layout** (best practice)
- ✅ **PEP 517/518/621** (modern standards)

See [SETUP_VS_PYPROJECT.md](SETUP_VS_PYPROJECT.md) for details.
