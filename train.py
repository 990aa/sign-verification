"""
Training script for Siamese Network signature verification
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from siamese_network import SiameseNetwork, ContrastiveLoss
from dataset import create_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_distance_genuine = 0.0
    running_distance_forgery = 0.0
    num_genuine = 0
    num_forgery = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for img1, img2, labels in pbar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        emb1, emb2 = model(img1, img2)
        loss, distances = criterion(emb1, emb2, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Track distances for genuine and forgery pairs
        genuine_mask = labels == 1
        forgery_mask = labels == 0
        
        if genuine_mask.sum() > 0:
            running_distance_genuine += distances[genuine_mask].mean().item() * genuine_mask.sum().item()
            num_genuine += genuine_mask.sum().item()
        
        if forgery_mask.sum() > 0:
            running_distance_forgery += distances[forgery_mask].mean().item() * forgery_mask.sum().item()
            num_forgery += forgery_mask.sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    avg_dist_genuine = running_distance_genuine / num_genuine if num_genuine > 0 else 0
    avg_dist_forgery = running_distance_forgery / num_forgery if num_forgery > 0 else 0
    
    return epoch_loss, avg_dist_genuine, avg_dist_forgery


def validate(model, dataloader, criterion, device, threshold=1.0):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_distance_genuine = 0.0
    running_distance_forgery = 0.0
    num_genuine = 0
    num_forgery = 0
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Validation"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Forward pass
            emb1, emb2 = model(img1, img2)
            loss, distances = criterion(emb1, emb2, labels)
            
            running_loss += loss.item()
            
            # Track distances
            genuine_mask = labels == 1
            forgery_mask = labels == 0
            
            if genuine_mask.sum() > 0:
                running_distance_genuine += distances[genuine_mask].mean().item() * genuine_mask.sum().item()
                num_genuine += genuine_mask.sum().item()
            
            if forgery_mask.sum() > 0:
                running_distance_forgery += distances[forgery_mask].mean().item() * forgery_mask.sum().item()
                num_forgery += forgery_mask.sum().item()
            
            # Calculate accuracy based on threshold
            predictions = (distances < threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    avg_dist_genuine = running_distance_genuine / num_genuine if num_genuine > 0 else 0
    avg_dist_forgery = running_distance_forgery / num_forgery if num_forgery > 0 else 0
    accuracy = 100 * correct / total
    
    return epoch_loss, avg_dist_genuine, avg_dist_forgery, accuracy


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Distance for genuine pairs
    axes[1, 0].plot(history['train_dist_genuine'], label='Train Genuine Distance')
    axes[1, 0].plot(history['val_dist_genuine'], label='Val Genuine Distance')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Distance')
    axes[1, 0].set_title('Average Distance for Genuine Pairs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Distance for forgery pairs
    axes[1, 1].plot(history['train_dist_forgery'], label='Train Forgery Distance')
    axes[1, 1].plot(history['val_dist_forgery'], label='Val Forgery Distance')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].set_title('Average Distance for Forgery Pairs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")


def train_model(dataset_path, num_epochs=50, batch_size=32, learning_rate=0.001, 
                embedding_dim=128, margin=2.0, save_dir='models'):
    """
    Main training function
    
    Args:
        dataset_path: Path to CEDAR dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        embedding_dim: Dimension of embedding vector
        margin: Margin for contrastive loss
        save_dir: Directory to save models and plots
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(dataset_path, batch_size=batch_size)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'train_dist_genuine': [],
        'train_dist_forgery': [],
        'val_dist_genuine': [],
        'val_dist_forgery': []
    }
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_dist_genuine, train_dist_forgery = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dist_genuine, val_dist_forgery, val_accuracy = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['train_dist_genuine'].append(train_dist_genuine)
        history['train_dist_forgery'].append(train_dist_forgery)
        history['val_dist_genuine'].append(val_dist_genuine)
        history['val_dist_forgery'].append(val_dist_forgery)
        
        # Print statistics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")
        print(f"  Train Dist (Genuine/Forgery): {train_dist_genuine:.4f} / {train_dist_forgery:.4f}")
        print(f"  Val Dist (Genuine/Forgery): {val_dist_genuine:.4f} / {val_dist_forgery:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (loss: {val_loss:.4f})")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, save_dir / 'best_accuracy_model.pth')
            print(f"  ✓ Saved best accuracy model ({val_accuracy:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, save_dir / 'final_model.pth')
    print(f"\n✓ Final model saved")
    
    # Plot training history
    plot_training_history(history, save_dir / 'training_history.png')
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Models saved in: {save_dir}")
    print(f"{'='*60}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Train Siamese Network for Signature Verification')
    parser.add_argument('dataset_path', nargs='?', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    
    # Get dataset path from command line or use default
    if not dataset_path:
        # Try to find the dataset in common locations
        possible_paths = [
            "cedar_signatures",
            "cedar-signatures",
            "data/cedar-signatures",
            "/root/.cache/kagglehub/datasets/matteocarnebella/cedar-signatures/versions/1"
        ]
        
        dataset_path = None
        for path in possible_paths:
            if Path(path).exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            print("Error: Dataset path not found!")
            print("Please run main.py first to download the dataset, or specify path:")
            print("  uv run python train.py /path/to/cedar-signatures")
            sys.exit(1)
    
    print(f"Using dataset from: {dataset_path}")
    
    # Train the model
    model, history = train_model(
        dataset_path=dataset_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=0.001,
        embedding_dim=128,
        margin=2.0
    )
