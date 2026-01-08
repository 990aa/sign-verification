"""
Download CEDAR Signature Dataset
This script downloads the dataset needed for signature verification
"""

import kagglehub
from pathlib import Path

print("="*70)
print(" Downloading CEDAR Signature Dataset")
print("="*70)
print("\nThis will download the CEDAR signature dataset from Kaggle.")
print("The dataset contains genuine signatures and forgeries from 55 writers.")
print()

# Download the dataset
path = kagglehub.dataset_download("matteocarnebella/cedar-signatures")

print(f"\n✓ Dataset downloaded successfully!")
print(f"📁 Dataset location: {path}")
print()

# Verify the dataset structure
dataset_path = Path(path)
if dataset_path.exists():
    genuine_dir = dataset_path / "full_org"
    forgery_dir = dataset_path / "full_forg"
    
    if genuine_dir.exists() and forgery_dir.exists():
        genuine_count = len(list(genuine_dir.glob("*.png")))
        forgery_count = len(list(forgery_dir.glob("*.png")))
        
        print("Dataset Structure:")
        print(f"  ├── full_org/     ({genuine_count} genuine signatures)")
        print(f"  └── full_forg/    ({forgery_count} forged signatures)")
        print()
        print("="*70)
        print(" Next Steps")
        print("="*70)
        print("1. Train the model:")
        print(f"   uv run python train.py {path}")
        print()
        print("2. Run inference:")
        print("   uv run python inference.py <reference_sig> <query_sig>")
        print()
    else:
        print("⚠️  Warning: Expected directory structure not found")
        print(f"    Looking for: {genuine_dir} and {forgery_dir}")
else:
    print("⚠️  Warning: Dataset path does not exist")

print("="*70)