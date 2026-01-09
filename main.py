"""
Download CEDAR Signature Dataset
This script downloads the dataset needed for signature verification
"""

import kagglehub
import shutil
from pathlib import Path

print("=" * 70)
print(" Downloading CEDAR Signature Dataset")
print("=" * 70)
print("\nThis will download the CEDAR signature dataset from Kaggle.")
print("The dataset contains genuine signatures and forgeries from 55 writers.")
print()

# Download the dataset
path = kagglehub.dataset_download("matteocarnebella/cedar-signatures")

print(f"\n✓ Dataset downloaded successfully to cache: {path}")

# Define target directory in the current workspace
target_dir = Path("cedar_signatures")

# Copy the dataset to the current directory if it doesn't already exist
if not target_dir.exists():
    print(f"Copying dataset to local directory: {target_dir}")
    shutil.copytree(path, target_dir)
else:
    print(f"Local directory {target_dir} already exists.")

print(f"Dataset is ready at: {target_dir.absolute()}")
print()

# Verify the dataset structure
dataset_path = target_dir
if dataset_path.exists():
    genuine_dir = dataset_path / "full_org"
    forgery_dir = dataset_path / "full_forg"
    signatures_dir = dataset_path / "signatures"

    has_flat = genuine_dir.exists() and forgery_dir.exists()
    has_grouped = signatures_dir.exists() and any(signatures_dir.glob("signatures_*"))

    if has_flat or has_grouped:
        if has_flat:
            genuine_count = len(list(genuine_dir.glob("*.png")))
            forgery_count = len(list(forgery_dir.glob("*.png")))
        else:
            matched_files = list(signatures_dir.rglob("*.png"))
            genuine_count = sum(1 for p in matched_files if "original" in p.name)
            forgery_count = sum(1 for p in matched_files if "forgeries" in p.name)

        print("\n✓ Dataset verification successful.")
        print(f"Found {genuine_count} genuine and {forgery_count} forgery signatures.")

        print("1. Train the model:")
        print(f"   uv run python train.py {dataset_path}")
        print()
        print("2. Run inference:")
        print("   uv run python inference.py <reference_sig> <query_sig>")
        print()
    else:
        print("Warning: Expected directory structure not found")
        print(f"    Looking for: {genuine_dir} and {forgery_dir}")
        print(f"    OR: {signatures_dir} with 'signatures_X' subfolders")
else:
    print("Warning: Dataset path does not exist")

print("=" * 70)
