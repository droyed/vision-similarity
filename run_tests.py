#!/usr/bin/env python3
"""
Run all test scripts in the tests/ directory.

This is a cross-platform test runner that works on Linux, macOS, and Windows.
"""

import subprocess
import sys
from pathlib import Path

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{BOLD}{'=' * 50}")
    print(f"{text}")
    print(f"{'=' * 50}{RESET}\n")

def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")

def main():
    """Run all test scripts."""
    print_header("Vision Similarity - Test Suite Runner")

    # Check if tests directory exists
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print_error("tests/ directory not found")
        print("Please run this script from the project root")
        sys.exit(1)

    # Check if package is installed
    try:
        import vision_similarity
        print(f"Package installed: vision_similarity v{vision_similarity.__version__}")
    except ImportError:
        print_error("vision_similarity package not installed")
        print("Please run: pip install -e .")
        sys.exit(1)

    # Test files to run
    tests = [
        "test_list_models.py",
        "test_vision_similarity.py",
        "test_plot_matrix.py",
        "test_similarity_workflows.py",
        "test_generate_feature_similarity_heatmaps.py",
    ]

    passed = 0
    failed = 0
    failed_tests = []

    # Run each test
    for test in tests:
        test_path = tests_dir / test

        if not test_path.exists():
            print_error(f"Test file not found: {test}")
            failed += 1
            failed_tests.append(test)
            continue

        print_header(f"Running: {test}")

        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            print_success(f"{test} passed")
            passed += 1
        else:
            print_error(f"{test} failed")
            failed += 1
            failed_tests.append(test)

    # Print summary
    print_header("Test Summary")
    print(f"Passed: {GREEN}{passed}{RESET}")
    print(f"Failed: {RED}{failed}{RESET}")

    if failed > 0:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        sys.exit(1)
    else:
        print_success("\nAll tests completed successfully! 🎉")
        sys.exit(0)

if __name__ == "__main__":
    main()
