# Signature Verification with Explainable AI

An advanced Offline Signature Verification system using Siamese Neural Networks with Grad-CAM explainability. It verifies if a signature is genuine or forged and visualizes the decision process.

## Theoretical Background

### Siamese Neural Networks
This project utilizes a Siamese Network architecture, which consists of two identical subnetworks that share the same weights and parameters. This architecture is particularly effective for signature verification because it learns a similarity metric rather than classifying inputs directly.
- **Twin Encoders**: The network takes a pair of signature images (reference and query) and processes them through identical CNN encoders.
- **Shared Weights**: Both encoders share the same weights, ensuring that two identical images map to the exact same feature vector.
- **Distance Metric**: The final layer computes the Euclidean distance between the two feature vectors to determine similarity.

### Contrastive Loss
To train the network, we employ Contrastive Loss. This loss function encourages the network to map genuine signature pairs close to each other in the embedding space while pushing forged pairs apart.
The loss function is defined as:
$$L = (1-Y)\frac{1}{2}D^2 + Y\frac{1}{2}\{\max(0, m-D)\}^2$$
Where $Y$ is the label (0 for genuine pair, 1 for forged pair), $D$ is the Euclidean distance, and $m$ is the margin.

### Attention Mechanisms
The network incorporates Attention Modules to enhance feature extraction:
- **Channel Attention**: Focuses on 'what' is meaningful in the input by recalibrating channel-wise feature responses.
- **Spatial Attention**: Focuses on 'where' the informative parts are, helping the network pay attention to important stroke details of the signature.

### Explainable AI with Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) is used to make the model's decisions transparent. It uses the gradients of the target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

## Key Features

1. Siamese Network Architecture
- Twin CNN-based encoders that learn signature embeddings
- Contrastive learning approach for robust verification
- CPU-optimized for fast inference

2. Explainable AI
- Grad-CAM visualization
- Heatmap overlays highlighting critical regions

3. Production-Ready Pipeline
- Complete training, validation, and inference scripts
- Comprehensive evaluation metrics
- Easy-to-use API for signature verification

## Quick Start

### Prerequisites
- Python 3.12+
- uv package manager

### Installation & Setup

1. Download the dataset:
```bash
uv run python main.py
```

2. Install dependencies: 
```bash
uv sync
```

3. Test enhancements:
```bash
uv run python test_enhancements.py
```

## Usage

### Training

Train the Siamese Network:
```bash
uv run python train.py
```
Or specify dataset path:
```bash
uv run python train.py /path/to/cedar-signatures
```

### Inference & Verification

Single Signature Verification:
```bash
uv run python inference.py reference_signature.png query_signature.png
```

Batch Verification:
```bash
uv run python inference.py reference.png query1.png query2.png query3.png
```

Demo Mode:
```bash
uv run python inference.py
```

Multi-Scale Grad-CAM:
```bash
uv run python inference.py reference.png query.png --multiscale
```
