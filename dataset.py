"""
Data loading and preprocessing for CEDAR signature dataset
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class SignatureDataset(Dataset):
    """
    Dataset for CEDAR signature verification
    Creates pairs of signatures (genuine-genuine and genuine-forgery)
    """
    
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir: Path to the dataset directory
            transform: Optional transform to be applied on images
            train: If True, use training mode (creates more pairs)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        
        # Load all signature images organized by writer
        self.signatures = {}  # writer_id -> list of (signature_path, is_genuine)
        self._load_signatures()
        
        # Create pairs
        self.pairs = []
        self.labels = []
        self._create_pairs()
        
    def _load_signatures(self):
        """Load all signature images and organize by writer ID"""
        # CEDAR dataset structure: full_org/original_X_Y.png and full_forg/forgeries_X_Y.png
        # where X is writer ID (1-55) and Y is signature number (1-24 for genuine, 1-24 for forgeries)
        
        genuine_dir = self.root_dir / "full_org"
        forgery_dir = self.root_dir / "full_forg"
        
        if not genuine_dir.exists() or not forgery_dir.exists():
            raise ValueError(f"Dataset directories not found in {self.root_dir}")
        
        # Load genuine signatures
        for img_path in genuine_dir.glob("original_*.png"):
            # Parse filename: original_X_Y.png
            filename = img_path.stem
            parts = filename.split("_")
            if len(parts) == 3:
                writer_id = int(parts[1])
                
                if writer_id not in self.signatures:
                    self.signatures[writer_id] = {'genuine': [], 'forgery': []}
                    
                self.signatures[writer_id]['genuine'].append(img_path)
        
        # Load forgery signatures
        for img_path in forgery_dir.glob("forgeries_*.png"):
            # Parse filename: forgeries_X_Y.png
            filename = img_path.stem
            parts = filename.split("_")
            if len(parts) == 3:
                writer_id = int(parts[1])
                
                if writer_id not in self.signatures:
                    self.signatures[writer_id] = {'genuine': [], 'forgery': []}
                    
                self.signatures[writer_id]['forgery'].append(img_path)
        
        print(f"Loaded signatures for {len(self.signatures)} writers")
        for writer_id in list(self.signatures.keys())[:3]:
            print(f"  Writer {writer_id}: {len(self.signatures[writer_id]['genuine'])} genuine, "
                  f"{len(self.signatures[writer_id]['forgery'])} forgeries")
    
    def _create_pairs(self):
        """Create pairs of signatures with labels"""
        self.pairs = []
        self.labels = []
        
        for writer_id, sigs in self.signatures.items():
            genuine_sigs = sigs['genuine']
            forgery_sigs = sigs['forgery']
            
            if len(genuine_sigs) < 2:
                continue
            
            # Create genuine pairs (label = 1)
            # Pair each genuine signature with other genuine signatures
            num_genuine_pairs = len(genuine_sigs) * 2  # Create multiple pairs per signature
            for _ in range(num_genuine_pairs):
                if len(genuine_sigs) >= 2:
                    img1, img2 = random.sample(genuine_sigs, 2)
                    self.pairs.append((img1, img2))
                    self.labels.append(1)
            
            # Create forgery pairs (label = 0)
            # Pair genuine signatures with forgeries
            if len(forgery_sigs) > 0:
                num_forgery_pairs = len(genuine_sigs) * 2
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
        img1 = Image.open(img1_path).convert('L')  # Convert to grayscale
        img2 = Image.open(img2_path).convert('L')
        
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
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform


def create_dataloaders(dataset_path, batch_size=32, train_split=0.8):
    """
    Create training and validation dataloaders
    
    Args:
        dataset_path: Path to the downloaded CEDAR dataset
        batch_size: Batch size for training
        train_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader
    """
    train_transform, val_transform = get_data_transforms()
    
    # Create full dataset
    full_dataset = SignatureDataset(dataset_path, transform=train_transform, train=True)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    # Note: This is a workaround since random_split doesn't allow different transforms
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader
