"""
Siamese Network Architecture for Signature Verification
Features a CNN-based embedding network with attention mechanism and contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Focuses on 'what' is meaningful given an input
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on 'where' is meaningful given an input
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(out))
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese Network with shared CNN encoder and attention mechanisms for signature verification.
    Uses contrastive learning to create embeddings where genuine signatures
    are close together and forgeries are far apart.
    Enhanced with CBAM attention modules for better feature extraction.
    """
    
    def __init__(self, embedding_dim=128, use_attention=True):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        
        # Shared CNN encoder with attention
        # Block 1: 155x220x1 -> 77x110x32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(32) if use_attention else nn.Identity()
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 2: 77x110x32 -> 38x55x64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(64) if use_attention else nn.Identity()
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 3: 38x55x64 -> 19x27x128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(128) if use_attention else nn.Identity()
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 4: 19x27x128 -> 9x13x256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cbam4 = CBAM(256) if use_attention else nn.Identity()
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Combine into encoder for compatibility
        self.encoder = nn.Sequential(
            self.block1, self.cbam1, self.pool1,
            self.block2, self.cbam2, self.pool2,
            self.block3, self.cbam3, self.pool3,
            self.block4, self.cbam4, self.pool4
        )
        
        # Calculate the flattened size: 9 * 13 * 256 = 29,952
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 13 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
    def forward_one(self, x):
        """Forward pass through one branch of the siamese network"""
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def forward(self, img1, img2):
        """
        Forward pass through both branches
        Returns embeddings for both images
        """
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        return emb1, emb2
    
    def get_embedding(self, x):
        """Get embedding for a single image (used for inference)"""
        return self.forward_one(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function
    Takes embeddings of two images and a label:
    - label = 1 means images are similar (genuine pair)
    - label = 0 means images are dissimilar (forgery pair)
    """
    
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, label):
        """
        Args:
            emb1: Embedding of first image (batch_size, embedding_dim)
            emb2: Embedding of second image (batch_size, embedding_dim)
            label: 1 for genuine pair, 0 for forgery pair (batch_size,)
        """
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss formula
        # For genuine pairs (label=1): minimize distance
        # For forgery pairs (label=0): maximize distance (up to margin)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss, euclidean_distance
