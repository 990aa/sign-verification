"""
Inference script for signature verification with Grad-CAM visualization
Enhanced with multi-reference signature support
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from siamese_network import SiameseNetwork
from gradcam import GradCAM, MultiScaleGradCAM


class SignatureVerifier:
    """
    Easy-to-use interface for signature verification with explainability
    """

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (None for auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize model
        self.model = SiameseNetwork(embedding_dim=128)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

        if "val_accuracy" in checkpoint:
            print(f"  Model validation accuracy: {checkpoint['val_accuracy']:.2f}%")

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model)
        self.multiscale_grad_cam = MultiScaleGradCAM(self.model)

        # Transform for preprocessing images
        self.transform = transforms.Compose(
            [
                transforms.Resize((155, 220)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        # Default threshold for classification
        self.threshold = 1.0

    def load_image(self, image_path):
        """Load and preprocess a signature image"""
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        return img_tensor

    def verify(
        self,
        reference_path,
        query_path,
        visualize=True,
        save_path=None,
        multiscale=False,
    ):
        """
        Verify if a query signature matches a reference signature

        Args:
            reference_path: Path to reference (known genuine) signature
            query_path: Path to query signature to verify
            visualize: If True, generate Grad-CAM visualization
            save_path: Optional path to save visualization
            multiscale: If True, use multi-scale Grad-CAM for detailed analysis

        Returns:
            dict with keys:
                - is_genuine: Boolean indicating if signatures match
                - distance: Euclidean distance between embeddings
                - confidence: Confidence score (0-100)
                - explanation: Human-readable explanation
        """
        # Load images
        img1 = self.load_image(reference_path).to(self.device)
        img2 = self.load_image(query_path).to(self.device)

        # Get embeddings and distance
        with torch.no_grad():
            emb1, emb2 = self.model(img1, img2)
            distance = torch.nn.functional.pairwise_distance(emb1, emb2).item()

        # Make decision
        is_genuine = distance < self.threshold

        # Calculate confidence (convert distance to 0-100 scale)
        # Closer distances = higher confidence for genuine
        # Further distances = higher confidence for forgery
        if is_genuine:
            confidence = max(0, min(100, (1 - distance / self.threshold) * 100))
        else:
            confidence = max(
                0, min(100, ((distance - self.threshold) / self.threshold) * 100)
            )

        # Generate explanation
        explanation = self._generate_explanation(distance, is_genuine, confidence)

        result = {
            "is_genuine": is_genuine,
            "distance": distance,
            "confidence": confidence,
            "explanation": explanation,
        }

        # Visualize with Grad-CAM
        if visualize:
            if multiscale:
                # Use multi-scale Grad-CAM for detailed analysis
                fig = self.multiscale_grad_cam.visualize_multiscale(
                    img1, img2, save_path=save_path
                )
            else:
                # Use standard Grad-CAM
                fig = self.grad_cam.visualize_comparison(
                    img1, img2, label=1 if is_genuine else 0, save_path=save_path
                )

            if save_path is None:
                plt.show()
            else:
                plt.close(fig)

        return result

    def verify_multi_reference(
        self,
        reference_paths,
        query_path,
        method="average",
        visualize=True,
        save_dir="results",
    ):
        """
        Verify a query signature against multiple reference signatures
        Provides more robust verification by comparing with multiple samples

        Args:
            reference_paths: List of paths to reference signatures
            query_path: Path to query signature to verify
            method: 'average' (mean distance), 'minimum' (best match), or 'voting'
            visualize: If True, generate visualizations
            save_dir: Directory to save visualizations

        Returns:
            dict with keys:
                - is_genuine: Boolean indicating if signatures match
                - distance: Combined distance score
                - individual_distances: List of distances from each reference
                - confidence: Confidence score (0-100)
                - explanation: Human-readable explanation
        """
        save_dir = Path(save_dir)
        if visualize:
            save_dir.mkdir(exist_ok=True)

        # Load query image
        query_img = self.load_image(query_path).to(self.device)

        # Verify against each reference
        distances = []
        decisions = []

        print("\n🔍 Multi-Reference Verification")
        print(f"Query: {query_path}")
        print(f"References: {len(reference_paths)} signatures")
        print("-" * 70)

        for i, ref_path in enumerate(reference_paths, 1):
            ref_img = self.load_image(ref_path).to(self.device)

            # Get embeddings and distance
            with torch.no_grad():
                emb1, emb2 = self.model(ref_img, query_img)
                distance = torch.nn.functional.pairwise_distance(emb1, emb2).item()

            distances.append(distance)
            decisions.append(distance < self.threshold)

            print(
                f"  [{i}/{len(reference_paths)}] {Path(ref_path).name}: "
                f"distance={distance:.4f} {'✓' if distance < self.threshold else '✗'}"
            )

            # Save individual visualization
            if visualize:
                save_path = save_dir / f"ref_{i}_comparison.png"
                self.grad_cam.visualize_comparison(
                    ref_img,
                    query_img,
                    label=1 if distance < self.threshold else 0,
                    save_path=save_path,
                )
                plt.close()

        # Combine results based on method
        if method == "average":
            combined_distance = np.mean(distances)
            is_genuine = combined_distance < self.threshold
            explanation_method = "average distance"
        elif method == "minimum":
            combined_distance = np.min(distances)
            is_genuine = combined_distance < self.threshold
            explanation_method = "minimum distance (best match)"
        elif method == "voting":
            is_genuine = sum(decisions) > len(decisions) / 2
            combined_distance = np.mean(distances)
            explanation_method = (
                f"majority voting ({sum(decisions)}/{len(decisions)} matches)"
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate confidence
        if is_genuine:
            confidence = max(
                0, min(100, (1 - combined_distance / self.threshold) * 100)
            )
        else:
            confidence = max(
                0,
                min(100, ((combined_distance - self.threshold) / self.threshold) * 100),
            )

        # Generate explanation
        explanation = self._generate_multi_ref_explanation(
            is_genuine,
            combined_distance,
            distances,
            decisions,
            method,
            explanation_method,
            confidence,
        )

        result = {
            "is_genuine": is_genuine,
            "distance": combined_distance,
            "individual_distances": distances,
            "individual_decisions": decisions,
            "confidence": confidence,
            "explanation": explanation,
            "method": method,
        }

        # Print summary
        print("\n" + "=" * 70)
        print("MULTI-REFERENCE VERIFICATION RESULT")
        print("=" * 70)
        print(explanation)
        print("=" * 70)

        return result

    def _generate_multi_ref_explanation(
        self,
        is_genuine,
        combined_distance,
        distances,
        decisions,
        method,
        explanation_method,
        confidence,
    ):
        """Generate human-readable explanation for multi-reference verification"""
        explanation = []

        if is_genuine:
            explanation.append("✓ GENUINE SIGNATURE DETECTED")
            explanation.append(f"  The signature matches based on {explanation_method}")
            explanation.append(f"  Confidence: {confidence:.1f}%")
            explanation.append(
                f"  Combined distance: {combined_distance:.4f} (threshold: {self.threshold:.2f})"
            )
        else:
            explanation.append("✗ FORGERY DETECTED")
            explanation.append(
                f"  The signature does not match based on {explanation_method}"
            )
            explanation.append(f"  Confidence: {confidence:.1f}%")
            explanation.append(
                f"  Combined distance: {combined_distance:.4f} (threshold: {self.threshold:.2f})"
            )

        explanation.append("\n  Individual Results:")
        for i, (dist, decision) in enumerate(zip(distances, decisions), 1):
            status = "✓ Match" if decision else "✗ No match"
            explanation.append(f"    Reference {i}: {dist:.4f} - {status}")

        explanation.append("\n  Statistics:")
        explanation.append(f"    Mean distance: {np.mean(distances):.4f}")
        explanation.append(f"    Min distance: {np.min(distances):.4f}")
        explanation.append(f"    Max distance: {np.max(distances):.4f}")
        explanation.append(f"    Std deviation: {np.std(distances):.4f}")
        explanation.append(f"    Matches: {sum(decisions)}/{len(decisions)}")

        return "\n".join(explanation)

    def batch_verify(self, reference_path, query_paths, save_dir="results"):
        """
        Verify multiple query signatures against a single reference

        Args:
            reference_path: Path to reference signature
            query_paths: List of paths to query signatures
            save_dir: Directory to save visualizations

        Returns:
            List of results for each query
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        results = []

        print(f"\nVerifying {len(query_paths)} signatures against reference...")
        print(f"Reference: {reference_path}")
        print("-" * 60)

        for i, query_path in enumerate(query_paths, 1):
            print(f"\n[{i}/{len(query_paths)}] Query: {query_path}")

            save_path = save_dir / f"verification_{i}.png"
            result = self.verify(
                reference_path, query_path, visualize=True, save_path=save_path
            )

            result["query_path"] = query_path
            results.append(result)

            # Print result
            status = "✓ GENUINE" if result["is_genuine"] else "✗ FORGERY"
            print(f"  Result: {status}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Confidence: {result['confidence']:.1f}%")

        # Summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        genuine_count = sum(1 for r in results if r["is_genuine"])
        forgery_count = len(results) - genuine_count
        print(f"Genuine signatures: {genuine_count}")
        print(f"Forgeries detected: {forgery_count}")
        print(f"Results saved to: {save_dir}")

        return results

    def _generate_explanation(self, distance, is_genuine, confidence):
        """Generate human-readable explanation"""
        explanation = []

        if is_genuine:
            explanation.append("✓ GENUINE SIGNATURE DETECTED")
            explanation.append(
                f"  The signatures match with {confidence:.1f}% confidence"
            )
            explanation.append(
                f"  Embedding distance: {distance:.4f} (threshold: {self.threshold:.2f})"
            )
            explanation.append(
                "  The model found consistent patterns between the signatures"
            )
        else:
            explanation.append("✗ FORGERY DETECTED")
            explanation.append(
                f"  The signatures do not match ({confidence:.1f}% confidence)"
            )
            explanation.append(
                f"  Embedding distance: {distance:.4f} (threshold: {self.threshold:.2f})"
            )
            explanation.append(
                "  The model detected significant differences in signature patterns"
            )

        return "\n".join(explanation)

    def set_threshold(self, threshold):
        """Set custom threshold for classification"""
        self.threshold = threshold
        print(f"Threshold updated to: {threshold:.2f}")


def main():
    """Example usage of the signature verifier"""
    import sys
    from pathlib import Path

    print("=" * 70)
    print(" Signature Verification System with Explainable AI (Grad-CAM)")
    print("=" * 70)

    # Check for model
    model_path = Path("models/best_accuracy_model.pth")
    if not model_path.exists():
        model_path = Path("models/best_model.pth")
    if not model_path.exists():
        model_path = Path("models/final_model.pth")

    if not model_path.exists():
        print("\nError: No trained model found!")
        print("Please run training first:")
        print("  uv run python train.py")
        sys.exit(1)

    # Initialize verifier
    verifier = SignatureVerifier(model_path)

    # Check command line arguments
    if len(sys.argv) == 3:
        # Two images provided
        reference_path = sys.argv[1]
        query_path = sys.argv[2]

        print("\nVerifying signatures:")
        print(f"  Reference: {reference_path}")
        print(f"  Query: {query_path}")
        print("-" * 70)

        result = verifier.verify(
            reference_path,
            query_path,
            visualize=True,
            save_path="verification_result.png",
        )

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(result["explanation"])
        print("=" * 70)
        print("\nVisualization saved to: verification_result.png")

    elif len(sys.argv) > 3:
        # Multiple queries against one reference
        reference_path = sys.argv[1]
        query_paths = sys.argv[2:]

        _ = verifier.batch_verify(reference_path, query_paths)

    else:
        # No arguments - show usage and try to find sample images
        print("\nUsage:")
        print("  Single verification:")
        print("    uv run python inference.py <reference_signature> <query_signature>")
        print("\n  Batch verification:")
        print("    uv run python inference.py <reference> <query1> <query2> ...")
        print("\nExample:")
        print("  uv run python inference.py genuine1.png genuine2.png")

        # Try to find dataset and run demo
        dataset_paths = [
            "/root/.cache/kagglehub/datasets/matteocarnebella/cedar-signatures/versions/1",
            "cedar-signatures",
            "data/cedar-signatures",
        ]

        dataset_path = None
        for path in dataset_paths:
            if Path(path).exists():
                dataset_path = Path(path)
                break

        if dataset_path:
            print(f"\n✓ Found dataset at: {dataset_path}")
            print("Running demo with sample signatures...")

            # Find some sample images
            genuine_dir = dataset_path / "full_org"
            forgery_dir = dataset_path / "full_forg"

            if genuine_dir.exists() and forgery_dir.exists():
                # Get a few samples
                genuine_samples = sorted(list(genuine_dir.glob("original_1_*.png")))[:2]
                forgery_samples = sorted(list(forgery_dir.glob("forgeries_1_*.png")))[
                    :1
                ]

                if len(genuine_samples) >= 2:
                    print("\n" + "=" * 70)
                    print("DEMO: Genuine vs Genuine")
                    print("=" * 70)
                    result = verifier.verify(
                        str(genuine_samples[0]),
                        str(genuine_samples[1]),
                        visualize=True,
                        save_path="demo_genuine.png",
                    )
                    print(result["explanation"])

                if len(genuine_samples) >= 1 and len(forgery_samples) >= 1:
                    print("\n" + "=" * 70)
                    print("DEMO: Genuine vs Forgery")
                    print("=" * 70)
                    result = verifier.verify(
                        str(genuine_samples[0]),
                        str(forgery_samples[0]),
                        visualize=True,
                        save_path="demo_forgery.png",
                    )
                    print(result["explanation"])

                print("\n✓ Demo visualizations saved!")


if __name__ == "__main__":
    main()
