#!/usr/bin/env python3
"""
System verification script - checks that all components are properly installed
"""

import sys
from pathlib import Path


def check_file(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ {description}: {filepath} NOT FOUND")
        return False


def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"  ✓ {description}")
        return True
    except ImportError as e:
        print(f"  ✗ {description} - {e}")
        return False


def main():
    print("=" * 70)
    print(" Signature Verification System - Installation Check")
    print("=" * 70)

    all_checks_passed = True

    # Check Python files
    print("\nChecking Project Files...")
    files = {
        "main.py": "Dataset downloader",
        "siamese_network.py": "Network architecture",
        "dataset.py": "Data pipeline",
        "train.py": "Training script",
        "gradcam.py": "Grad-CAM explainability",
        "inference.py": "Inference API",
        "quickstart.py": "Quick start guide",
        "pyproject.toml": "Project configuration",
        "README.md": "Documentation",
    }

    for file, desc in files.items():
        if not check_file(file, desc):
            all_checks_passed = False

    print("\nChecking Dependencies...")
    dependencies = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "matplotlib": "Matplotlib",
        "cv2": "OpenCV",
        "sklearn": "Scikit-learn",
        "tqdm": "TQDM",
        "kagglehub": "KaggleHub",
    }

    for module, desc in dependencies.items():
        if not check_import(module, desc):
            all_checks_passed = False

    print("\nChecking Python Version...")
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 12:
        print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(
            f"  ✗ Python {py_version.major}.{py_version.minor}.{py_version.micro} (need 3.12+)"
        )
        all_checks_passed = False

    print("\nChecking Dataset...")
    dataset_paths = [
        "/root/.cache/kagglehub/datasets/matteocarnebella/cedar-signatures/versions/1",
        "cedar-signatures",
        "data/cedar-signatures",
    ]

    dataset_found = False
    for path in dataset_paths:
        if Path(path).exists():
            print(f"  ✓ Dataset found at: {path}")
            dataset_found = True
            break

    if not dataset_found:
        print("  Dataset not found (run: uv run python main.py)")

    # Check for trained model
    print("\nChecking Trained Model...")
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            print(f"  ✓ Found {len(model_files)} model file(s):")
            for model in model_files:
                print(f"      • {model.name}")
        else:
            print("  No trained models found (run: uv run python train.py)")
    else:
        print("  Models directory not found (run: uv run python train.py)")

    # Summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print(" ✅ All checks passed!")
        print("=" * 70)
        print("\n🚀 Your system is ready!")
        print("\nNext steps:")
        if not dataset_found:
            print("  1. Download dataset: uv run python main.py")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            print("  2. Train model: uv run python train.py")
        print("  3. Run inference: uv run python inference.py")
        print("\nOr use the guided setup:")
        print("  uv run python quickstart.py")
    else:
        print(" Some checks failed!")
        print("=" * 70)
        print("\nTo fix issues:")
        print("  1. Install dependencies: uv sync")
        print("  2. Check Python version: python --version (need 3.12+)")
        print("  3. Verify files are present")

    print("\n" + "=" * 70)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
