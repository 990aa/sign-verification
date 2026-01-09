import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")


def print_step(number, text):
    """Print step number"""
    print(f"\n{'=' * 70}")
    print(f"STEP {number}: {text}")
    print("=" * 70)


def check_dataset():
    """Check if dataset exists"""
    possible_paths = [
        "/root/.cache/kagglehub/datasets/matteocarnebella/cedar-signatures/versions/1",
        "cedar-signatures",
        "data/cedar-signatures",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path
    return None


def check_model():
    """Check if trained model exists"""
    model_dir = Path("models")
    if not model_dir.exists():
        return None

    for model_name in ["best_accuracy_model.pth", "best_model.pth", "final_model.pth"]:
        model_path = model_dir / model_name
        if model_path.exists():
            return model_path

    return None


def main():
    print_header("Signature Verification System - Quick Start")

    print("This script will guide you through:")
    print("  1. Downloading the CEDAR signature dataset")
    print("  2. Training the Siamese Network")
    print("  3. Running inference with Grad-CAM visualization")
    print()
    input("Press Enter to continue...")

    # Step 1: Check/Download Dataset
    print_step(1, "Dataset Setup")

    dataset_path = check_dataset()
    if dataset_path:
        print(f"✓ Dataset already exists at: {dataset_path}")
    else:
        print("Dataset not found. Downloading...")
        print("\nRunning: uv run python main.py")
        print()
        import subprocess

        result = subprocess.run(["uv", "run", "python", "main.py"])

        if result.returncode != 0:
            print("\nError downloading dataset!")
            sys.exit(1)

        # Check again
        dataset_path = check_dataset()
        if not dataset_path:
            print("\nDataset download failed!")
            sys.exit(1)

    # Step 2: Check/Train Model
    print_step(2, "Model Training")

    model_path = check_model()
    if model_path:
        print(f"✓ Trained model already exists: {model_path}")
        print("\nSkipping training. To retrain, delete the models/ directory.")
    else:
        print("No trained model found. Starting training...")
        print("\nNote: Training may take 30-60 minutes depending on your hardware.")
        print("You can interrupt training with Ctrl+C and models will be saved.")
        print()

        response = input("Start training now? (y/N): ")
        if response.lower() != "y":
            print("\nTraining skipped. You can train later with:")
            print(f"  uv run python train.py {dataset_path}")
            print("\nNote: You need a trained model to run inference.")
            sys.exit(0)

        print("\nRunning: uv run python train.py")
        print()
        import subprocess

        result = subprocess.run(["uv", "run", "python", "train.py", dataset_path])

        if result.returncode != 0:
            print("\nTraining failed or was interrupted!")
            print("You can resume training later with:")
            print(f"  uv run python train.py {dataset_path}")
            sys.exit(1)

        # Check for model
        model_path = check_model()
        if not model_path:
            print("\nModel file not found after training!")
            sys.exit(1)

    # Step 3: Run Inference Demo
    print_step(3, "Inference Demo")

    print("Running demo inference with Grad-CAM visualization...")
    print("\nThis will:")
    print("  • Compare genuine signatures (should match)")
    print("  • Compare genuine vs forgery (should not match)")
    print("  • Generate visualization with heatmaps")
    print()

    print("Running: uv run python inference.py")
    print()
    import subprocess

    result = subprocess.run(["uv", "run", "python", "inference.py"])

    if result.returncode != 0:
        print("\nDemo had issues, but you can still use the system manually:")
        print("\n  uv run python inference.py <reference_sig> <query_sig>")

    print_header("Setup Complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
