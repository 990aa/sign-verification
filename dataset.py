"""
Data loading and preprocessing for CEDAR signature dataset
"""

import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SignatureDataset(Dataset):
    """
    Dataset for CEDAR signature verification
    Creates pairs of signatures (genuine-genuine and genuine-forgery)
    """

    def __init__(self, root_dir, writer_ids=None, transform=None, train=True):
        """
        Args:
            root_dir: Path to the dataset directory
            writer_ids: List of writer IDs to include. If None, include all.
            transform: Optional transform to be applied on images
            train: If True, use training mode (creates more pairs)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train

        # Load all signature images organized by writer
        self.signatures = {}  # writer_id -> list of (signature_path, is_genuine)
        self._load_signatures(writer_ids)

        # Create pairs
        self.pairs = []
        self.labels = []
        self._create_pairs()

    def _load_signatures(self, allowed_writer_ids=None):
        """Load all signature images and organize by writer ID
        Supports both flat (full_org/full_forg) and grouped (signatures/signatures_X) structures
        """
        # 1. Try flat structure: root/full_org and root/full_forg
        genuine_flat = self.root_dir / "full_org"
        forgery_flat = self.root_dir / "full_forg"

        # 2. Try grouped structure: root/signatures/signatures_X
        grouped_candidates = [self.root_dir / "signatures", self.root_dir]

        if genuine_flat.exists() and forgery_flat.exists():
            print(f"Found flat dataset structure in {self.root_dir}")
            self._load_from_dirs(genuine_flat, forgery_flat, allowed_writer_ids)
        else:
            # Check for grouped structure
            found_grouped = False
            for base_dir in grouped_candidates:
                if base_dir.exists() and list(base_dir.glob("signatures_*")):
                    print(f"Found grouped dataset structure in {base_dir}")
                    found_grouped = True
                    for author_dir in base_dir.glob("signatures_*"):
                        self._load_from_mixed_dir(author_dir, allowed_writer_ids)
                    break

            if not found_grouped:
                # Debug info
                print(f"Checked in {self.root_dir}")
                print(f"  Gap: {genuine_flat.exists()}, {forgery_flat.exists()}")
                print(f"  Grouped checks: {[d.exists() for d in grouped_candidates]}")
                raise ValueError(
                    f"Dataset directories not found in {self.root_dir}. Expected 'full_org'/'full_forg' OR 'signatures/signatures_X' structure."
                )

        print(f"Loaded signatures for {len(self.signatures)} writers")
        if self.signatures:
            for writer_id in list(self.signatures.keys())[:3]:
                print(
                    f"  Writer {writer_id}: {len(self.signatures[writer_id]['genuine'])} genuine, "
                    f"{len(self.signatures[writer_id]['forgery'])} forgeries"
                )

    def _load_from_dirs(self, genuine_dir, forgery_dir, allowed_writer_ids=None):
        # Load genuine
        for img_path in genuine_dir.glob("original_*.png"):
            self._add_image(
                img_path, is_genuine=True, allowed_writer_ids=allowed_writer_ids
            )
        # Load forgery
        for img_path in forgery_dir.glob("forgeries_*.png"):
            self._add_image(
                img_path, is_genuine=False, allowed_writer_ids=allowed_writer_ids
            )

    def _load_from_mixed_dir(self, directory, allowed_writer_ids=None):
        # Load genuine
        for img_path in directory.glob("original_*.png"):
            self._add_image(
                img_path, is_genuine=True, allowed_writer_ids=allowed_writer_ids
            )
        # Load forgery
        for img_path in directory.glob("forgeries_*.png"):
            self._add_image(
                img_path, is_genuine=False, allowed_writer_ids=allowed_writer_ids
            )

    def _add_image(self, img_path, is_genuine, allowed_writer_ids=None):
        filename = img_path.stem
        parts = filename.split("_")
        if len(parts) >= 3:
            try:
                writer_id = int(parts[1])

                # Filter by writer ID if specified
                if (
                    allowed_writer_ids is not None
                    and writer_id not in allowed_writer_ids
                ):
                    return

                if writer_id not in self.signatures:
                    self.signatures[writer_id] = {"genuine": [], "forgery": []}

                key = "genuine" if is_genuine else "forgery"
                self.signatures[writer_id][key].append(img_path)
            except ValueError:
                pass

    def _create_pairs(self):
        """Create pairs of signatures with labels"""
        self.pairs = []
        self.labels = []

        for writer_id, sigs in self.signatures.items():
            genuine_sigs = sigs["genuine"]
            forgery_sigs = sigs["forgery"]

            if len(genuine_sigs) < 2:
                continue

            # Create genuine pairs (label = 1)
            # Pair each genuine signature with other genuine signatures
            # Increase pair generation for better training
            pairs_per_sig = 5 if self.train else 2
            num_genuine_pairs = len(genuine_sigs) * pairs_per_sig

            for _ in range(num_genuine_pairs):
                if len(genuine_sigs) >= 2:
                    img1, img2 = random.sample(genuine_sigs, 2)
                    self.pairs.append((img1, img2))
                    self.labels.append(1)

            # Create forgery pairs (label = 0)
            # Pair genuine signatures with forgeries
            if len(forgery_sigs) > 0:
                num_forgery_pairs = len(genuine_sigs) * pairs_per_sig
                for _ in range(num_forgery_pairs):
                    img1 = random.choice(genuine_sigs)
                    img2 = random.choice(forgery_sigs)
                    self.pairs.append((img1, img2))
                    self.labels.append(0)

        print(f"Created {len(self.pairs)} pairs:")
        print(f"  Genuine pairs (label=1): {sum(self.labels)}")
        print(f"  Forgery pairs (label=0): {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            img1: First signature image
            img2: Second signature image
            label: 1 if genuine pair, 0 if forgery pair
        """
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        # Load images
        img1 = Image.open(img1_path).convert("L")  # Convert to grayscale
        img2 = Image.open(img2_path).convert("L")

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_data_transforms(img_size=(155, 220)):
    """
    Get data transforms for training and validation
    CEDAR signatures are typically 155x220 pixels
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(dataset_path, batch_size=32, train_split=0.8):
    """
    Create training and validation dataloaders
    Uses writer-independent splitting
    """
    train_transform, val_transform = get_data_transforms()

    # helper to find all writers
    temp_dataset = SignatureDataset(dataset_path, train=False)
    all_writers = list(temp_dataset.signatures.keys())

    # shuffle and split writers
    random.seed(42)
    random.shuffle(all_writers)

    split_idx = int(len(all_writers) * train_split)
    train_writers = all_writers[:split_idx]
    val_writers = all_writers[split_idx:]

    # Create datasets with writer splits
    train_dataset = SignatureDataset(
        dataset_path, writer_ids=train_writers, transform=train_transform, train=True
    )

    val_dataset = SignatureDataset(
        dataset_path, writer_ids=val_writers, transform=val_transform, train=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
