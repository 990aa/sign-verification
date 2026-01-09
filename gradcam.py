"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI
Shows which parts of the signature contributed to the similarity/difference decision
Enhanced with multi-scale visualization for finer detail analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2


class MultiScaleGradCAM:
    """
    Multi-scale Grad-CAM implementation
    Generates heatmaps at different convolutional layers for finer detail analysis
    """

    def __init__(self, model):
        """
        Args:
            model: Trained Siamese Network
        """
        self.model = model
        self.model.eval()

        # Find all conv layers in the encoder for multi-scale analysis
        self.target_layers = []
        self.layer_names = []

        for idx, module in enumerate(self.model.encoder.modules()):
            if isinstance(module, torch.nn.Conv2d):
                self.target_layers.append(module)
                self.layer_names.append(f"Conv_{len(self.target_layers)}")

        print(
            f"Found {len(self.target_layers)} convolutional layers for multi-scale Grad-CAM"
        )

        self.activations = {}
        self.gradients = {}

        # Register hooks for all target layers
        for idx, layer in enumerate(self.target_layers):
            layer.register_forward_hook(self._make_save_activation(idx))
            layer.register_full_backward_hook(self._make_save_gradient(idx))

    def _make_save_activation(self, idx):
        """Create hook function to save activations"""

        def hook(module, input, output):
            self.activations[idx] = output.detach()

        return hook

    def _make_save_gradient(self, idx):
        """Create hook function to save gradients"""

        def hook(module, grad_input, grad_output):
            self.gradients[idx] = grad_output[0].detach()

        return hook

    def generate_multiscale_cam(self, img):
        """
        Generate multi-scale CAM for a single image
        Returns CAMs from different layers for detailed analysis
        """
        self.model.zero_grad()
        img.requires_grad = True

        # Forward pass
        emb = self.model.forward_one(img)
        score = torch.norm(emb, dim=1)
        score.backward()

        # Generate CAMs from all layers
        cams = []
        for idx in range(len(self.target_layers)):
            if idx in self.gradients and idx in self.activations:
                cam = self._generate_cam_from_layer(idx)
                cams.append(cam)

        return cams

    def _generate_cam_from_layer(self, layer_idx):
        """Generate CAM from a specific layer"""
        gradients = self.gradients[layer_idx]
        activations = self.activations[layer_idx]

        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize_multiscale(self, img1, img2, save_path=None):
        """
        Visualize multi-scale CAMs showing different levels of detail
        """
        # Generate multi-scale CAMs
        cams1 = self.generate_multiscale_cam(img1)
        cams2 = self.generate_multiscale_cam(img2)

        # Select key layers to visualize (early, middle, late)
        num_layers = len(cams1)
        if num_layers >= 3:
            selected_indices = [0, num_layers // 2, num_layers - 1]
            selected_names = ["Early (Edges)", "Middle (Patterns)", "Late (Semantics)"]
        else:
            selected_indices = list(range(num_layers))
            selected_names = [f"Layer {i + 1}" for i in selected_indices]

        # Convert images
        img1_np = img1.squeeze().cpu().numpy()
        img2_np = img2.squeeze().cpu().numpy()
        img1_np = (img1_np * 0.5 + 0.5).clip(0, 1)
        img2_np = (img2_np * 0.5 + 0.5).clip(0, 1)

        # Create visualization
        fig, axes = plt.subplots(
            len(selected_indices) + 1, 2, figsize=(12, 4 * (len(selected_indices) + 1))
        )

        fig.suptitle("Multi-Scale Grad-CAM Analysis", fontsize=16, fontweight="bold")

        # Original signatures
        axes[0, 0].imshow(img1_np, cmap="gray")
        axes[0, 0].set_title("Reference Signature", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(img2_np, cmap="gray")
        axes[0, 1].set_title("Query Signature", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # Multi-scale CAMs
        for idx, (layer_idx, name) in enumerate(zip(selected_indices, selected_names)):
            cam1 = cv2.resize(cams1[layer_idx], (img1_np.shape[1], img1_np.shape[0]))
            cam2 = cv2.resize(cams2[layer_idx], (img2_np.shape[1], img2_np.shape[0]))

            overlay1 = self._create_overlay(img1_np, cam1)
            overlay2 = self._create_overlay(img2_np, cam2)

            axes[idx + 1, 0].imshow(overlay1)
            axes[idx + 1, 0].set_title(f"{name} - Reference", fontsize=11)
            axes[idx + 1, 0].axis("off")

            axes[idx + 1, 1].imshow(overlay2)
            axes[idx + 1, 1].set_title(f"{name} - Query", fontsize=11)
            axes[idx + 1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Multi-scale visualization saved to {save_path}")

        return fig

    def _create_overlay(self, img, cam, alpha=0.5):
        """Create overlay of heatmap on grayscale image"""
        img_rgb = np.stack([img, img, img], axis=-1)
        colormap = plt.get_cmap("jet")
        cam_colored = colormap(cam)[:, :, :3]
        overlay = alpha * cam_colored + (1 - alpha) * img_rgb
        return np.clip(overlay, 0, 1)


class GradCAM:
    """
    Grad-CAM implementation for Siamese Network
    Generates heatmaps showing which parts of the signature the model focused on
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: Trained Siamese Network
            target_layer: The convolutional layer to compute gradients for
                         If None, uses the last conv layer
        """
        self.model = model
        self.model.eval()

        # If target layer not specified, use the last conv layer in the encoder
        if target_layer is None:
            # Find the last Conv2d layer in the encoder
            for module in reversed(list(self.model.encoder.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = module
                    break
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, img1, img2, return_distance=False):
        """
        Generate Grad-CAM heatmaps for both images

        Args:
            img1: First signature image tensor (1, 1, H, W)
            img2: Second signature image tensor (1, 1, H, W)
            return_distance: If True, also return the embedding distance

        Returns:
            cam1: Heatmap for first image
            cam2: Heatmap for second image
            distance: (optional) Euclidean distance between embeddings
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Forward pass to get embeddings
        img1.requires_grad = True
        img2.requires_grad = True

        emb1, emb2 = self.model(img1, img2)

        # Calculate distance
        distance = F.pairwise_distance(emb1, emb2)

        # For Grad-CAM, we want to see what contributes to similarity/dissimilarity
        # We backpropagate through the distance
        self.model.zero_grad()

        # Generate CAM for first image
        distance.backward(retain_graph=True)
        cam1 = self._generate_cam_from_gradients()

        # Clear gradients and generate CAM for second image
        self.model.zero_grad()

        # Need to recompute for second image
        # Re-run forward pass
        emb1, emb2 = self.model(img1, img2)
        distance = F.pairwise_distance(emb1, emb2)

        # This time backprop through second branch
        # We need a different approach - compute gradient for each branch separately
        cam1 = self._generate_cam_for_branch(img1)
        cam2 = self._generate_cam_for_branch(img2)

        if return_distance:
            return cam1, cam2, distance.item()
        return cam1, cam2

    def _generate_cam_for_branch(self, img):
        """Generate CAM for a single image"""
        self.model.zero_grad()

        # Forward pass through one branch
        emb = self.model.forward_one(img)

        # Backward pass - maximize the embedding magnitude
        # (we want to see what features the network extracted)
        score = torch.norm(emb, dim=1)
        score.backward()

        # Generate CAM
        return self._generate_cam_from_gradients()

    def _generate_cam_from_gradients(self):
        """Generate CAM from stored gradients and activations"""
        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)

        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)

        # Apply ReLU (only positive influences)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize_comparison(self, img1, img2, label=None, save_path=None):
        """
        Create a comprehensive visualization showing:
        - Original signatures
        - Grad-CAM heatmaps
        - Overlay of heatmap on signature

        Args:
            img1: First signature tensor (1, 1, H, W)
            img2: Second signature tensor (1, 1, H, W)
            label: Ground truth label (1=genuine, 0=forgery)
            save_path: Path to save the visualization
        """
        # Generate CAMs
        cam1, cam2, distance = self.generate_cam(img1, img2, return_distance=True)

        # Convert images to numpy for visualization
        img1_np = img1.squeeze().cpu().numpy()
        img2_np = img2.squeeze().cpu().numpy()

        # Denormalize images (assuming normalized with mean=0.5, std=0.5)
        img1_np = (img1_np * 0.5 + 0.5).clip(0, 1)
        img2_np = (img2_np * 0.5 + 0.5).clip(0, 1)

        # Resize CAMs to match image size
        cam1_resized = cv2.resize(cam1, (img1_np.shape[1], img1_np.shape[0]))
        cam2_resized = cv2.resize(cam2, (img2_np.shape[1], img2_np.shape[0]))

        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))

        # Title
        prediction = "GENUINE" if distance < 1.0 else "FORGERY"
        color = (
            "green"
            if (prediction == "GENUINE" and label == 1)
            or (prediction == "FORGERY" and label == 0)
            else "red"
        )

        title = f"Signature Verification - Distance: {distance:.3f}\n"
        title += f"Prediction: {prediction}"
        if label is not None:
            actual = "GENUINE" if label == 1 else "FORGERY"
            title += f" | Actual: {actual}"

        fig.suptitle(title, fontsize=14, fontweight="bold", color=color)

        # Row 1: Original signatures
        axes[0, 0].imshow(img1_np, cmap="gray")
        axes[0, 0].set_title("Reference Signature", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(img2_np, cmap="gray")
        axes[0, 1].set_title("Query Signature", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # Row 2: Grad-CAM heatmaps
        im1 = axes[1, 0].imshow(cam1_resized, cmap="jet")
        axes[1, 0].set_title("Attention Heatmap (Reference)", fontsize=11)
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        im2 = axes[1, 1].imshow(cam2_resized, cmap="jet")
        axes[1, 1].set_title("Attention Heatmap (Query)", fontsize=11)
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        # Row 3: Overlay
        overlay1 = self._create_overlay(img1_np, cam1_resized)
        overlay2 = self._create_overlay(img2_np, cam2_resized)

        axes[2, 0].imshow(overlay1)
        axes[2, 0].set_title("Overlay (Reference)", fontsize=11)
        axes[2, 0].axis("off")

        axes[2, 1].imshow(overlay2)
        axes[2, 1].set_title("Overlay (Query)", fontsize=11)
        axes[2, 1].axis("off")

        # Add explanation text
        explanation = self._generate_explanation(distance, cam1_resized, cam2_resized)
        fig.text(
            0.5,
            0.02,
            explanation,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            wrap=True,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        return fig

    def _create_overlay(self, img, cam, alpha=0.5):
        """Create overlay of heatmap on grayscale image"""
        # Convert grayscale to RGB
        img_rgb = np.stack([img, img, img], axis=-1)

        # Apply colormap to CAM
        colormap = plt.get_cmap("jet")
        cam_colored = colormap(cam)[:, :, :3]  # Remove alpha channel

        # Blend
        overlay = alpha * cam_colored + (1 - alpha) * img_rgb
        overlay = np.clip(overlay, 0, 1)

        return overlay

    def _generate_explanation(self, distance, cam1, cam2):
        """Generate human-readable explanation of the decision"""
        explanation = "🔍 Forensic Analysis:\n"

        # Decision
        if distance < 1.0:
            explanation += f"✓ Signatures MATCH (distance: {distance:.3f})\n"
        else:
            explanation += f"✗ Signatures DO NOT MATCH (distance: {distance:.3f})\n"

        # Analyze attention patterns
        # Find regions of high attention
        threshold = 0.7
        high_attention1 = (cam1 > threshold).sum() / cam1.size
        high_attention2 = (cam2 > threshold).sum() / cam2.size

        explanation += "Attention Analysis:\n"
        explanation += f"  • Reference signature: {high_attention1 * 100:.1f}% high-attention regions\n"
        explanation += f"  • Query signature: {high_attention2 * 100:.1f}% high-attention regions\n"

        # Interpretation
        if distance < 1.0:
            explanation += (
                "The model found consistent patterns between both signatures."
            )
        else:
            explanation += (
                "The model detected significant differences in signature patterns."
            )

        return explanation


def visualize_batch_predictions(model, dataloader, num_samples=5, save_dir="results"):
    """
    Visualize predictions for a batch of samples with Grad-CAM

    Args:
        model: Trained Siamese Network
        dataloader: DataLoader containing signature pairs
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    from pathlib import Path

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    device = next(model.parameters()).device
    grad_cam = GradCAM(model)

    # Get some samples
    dataiter = iter(dataloader)
    img1_batch, img2_batch, labels_batch = next(dataiter)

    for i in range(min(num_samples, len(labels_batch))):
        img1 = img1_batch[i : i + 1].to(device)
        img2 = img2_batch[i : i + 1].to(device)
        label = labels_batch[i].item()

        # Generate visualization
        save_path = (
            save_dir / f"sample_{i + 1}_{'genuine' if label == 1 else 'forgery'}.png"
        )
        grad_cam.visualize_comparison(img1, img2, label=label, save_path=save_path)
        plt.close()

    print(f"\n✓ Generated {num_samples} visualizations in {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("Grad-CAM module for signature verification explainability")
    print("Import this module and use GradCAM class to generate explanations")
