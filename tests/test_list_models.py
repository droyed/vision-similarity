"""
Demo script to test get_all_available_models_list and print_models_table functions.
"""

from vision_similarity.list_models import get_all_available_models_list, print_models_table
from vision_similarity.utils import login_huggingface_from_env
import warnings


# Login with huggingface token (optional)
login_huggingface_from_env()

VERIFY_AVAILABILITY = False
TABLE_FORMAT = 'fancy_outline'

if VERIFY_AVAILABILITY:
    try:
        login_huggingface_from_env()
    except Exception:
        VERIFY_AVAILABILITY = False
        warnings.warn("Forcing VERIFY_AVAILABILITY to False because login failed")

# Test 1: get_all_available_models_list
all_models = get_all_available_models_list(verify_availability=VERIFY_AVAILABILITY)
print(f"\nTotal models: {len(all_models)}")
for i, model in enumerate(all_models, 1):
    print(f"{i}. {model}")

# Test 2: print_models_table
print("\nPrinting models table...")
print_models_table(verify_availability=VERIFY_AVAILABILITY, tablefmt=TABLE_FORMAT)
