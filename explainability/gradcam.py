"""
Model Explainability for Bone Fracture Detection
Implements Grad-CAM, attention visualization, and other interpretability methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import logging

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not available. Install with: pip install grad-cam")


class FractureGradCAM:
    """Grad-CAM implementation for fracture detection models"""
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        use_cuda: bool = True
    ):
        """
        Args:
            model: PyTorch model
            target_layers: List of target layer names for visualization
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Store gradients and feature maps
        self.gradients = {}
        self.activations = {}
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layers):
                handle_forward = module.register_forward_hook(save_activation(name))
                handle_backward = module.register_backward_hook(save_gradient(name))
                self.hooks.extend([handle_forward, handle_backward])
                logging.info(f"Registered hooks for layer: {name}")
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM visualization
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for visualization (if None, use predicted class)
            layer_name: Specific layer name (if None, use first target layer)
            
        Returns:
            CAM heatmap as numpy array
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get the target layer
        if layer_name is None:
            layer_name = list(self.gradients.keys())[0]
        
        if layer_name not in self.gradients:
            raise ValueError(f"Layer {layer_name} not found in registered layers")
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        
        return cam
    
    def visualize_cam(
        self,
        image: Union[np.ndarray, Image.Image],
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay CAM on original image
        
        Args:
            image: Original image
            cam: CAM heatmap
            alpha: Overlay transparency
            colormap: Colormap for heatmap
            
        Returns:
            Overlaid image
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize CAM to image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        colormap_func = cm.get_cmap(colormap)
        cam_colored = colormap_func(cam_resized)[:, :, :3]  # Remove alpha channel
        
        # Normalize image to [0, 1]
        if image.max() > 1:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = image.astype(np.float32)
        
        # Overlay
        overlaid = (1 - alpha) * image_norm + alpha * cam_colored
        overlaid = np.clip(overlaid, 0, 1)
        
        return (overlaid * 255).astype(np.uint8)
    
    def cleanup(self):
        """Remove hooks to free memory"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class AdvancedExplainer:
    """Advanced explainability methods for fracture detection"""
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = 'auto'
    ):
        """
        Args:
            model: PyTorch model
            class_names: List of class names
            device: Device to use for computations
        """
        self.model = model
        self.class_names = class_names
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def get_target_layers(self) -> List[nn.Module]:
        """Get appropriate target layers for different architectures"""
        target_layers = []
        
        # For EfficientNet
        if hasattr(self.model, 'backbone') and 'efficientnet' in str(type(self.model.backbone)).lower():
            if hasattr(self.model.backbone, 'features'):
                target_layers.append(self.model.backbone.features[-1])
        
        # For ResNet
        elif hasattr(self.model, 'backbone') and 'resnet' in str(type(self.model.backbone)).lower():
            if hasattr(self.model.backbone, 'layer4'):
                target_layers.append(self.model.backbone.layer4[-1])
        
        # For DenseNet
        elif hasattr(self.model, 'backbone') and 'densenet' in str(type(self.model.backbone)).lower():
            if hasattr(self.model.backbone, 'features'):
                target_layers.append(self.model.backbone.features[-1])
        
        # Fallback: try to find last convolutional layer
        if not target_layers:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layers.append(module)
                    break
        
        return target_layers
    
    def generate_gradcam_explanations(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM explanations using pytorch-grad-cam
        
        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original image for overlay
            target_class: Target class (if None, use predicted)
            save_path: Path to save visualizations
            
        Returns:
            Dictionary with different CAM visualizations
        """
        if not GRAD_CAM_AVAILABLE:
            logging.error("pytorch-grad-cam not available")
            return {}
        
        input_tensor = input_tensor.to(self.device)
        target_layers = self.get_target_layers()
        
        if not target_layers:
            logging.error("No suitable target layers found")
            return {}
        
        results = {}
        
        # Get prediction if target_class not specified
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        if target_class is None:
            target_class = predicted_class
        
        target = [ClassifierOutputTarget(target_class)]
        
        # Generate different types of CAM
        cam_methods = {
            'GradCAM': GradCAM,
            'GradCAM++': GradCAMPlusPlus,
            'ScoreCAM': ScoreCAM,
            'LayerCAM': LayerCAM
        }
        
        for method_name, method_class in cam_methods.items():
            try:
                cam = method_class(model=self.model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)
                grayscale_cam = grayscale_cam[0, :]  # Take first image
                
                # Create visualization
                visualization = show_cam_on_image(
                    original_image / 255.0,
                    grayscale_cam,
                    use_rgb=True
                )
                
                results[method_name] = {
                    'heatmap': grayscale_cam,
                    'visualization': visualization,
                    'predicted_class': predicted_class,
                    'target_class': target_class,
                    'confidence': confidence
                }
                
            except Exception as e:
                logging.warning(f"Failed to generate {method_name}: {e}")
        
        # Save visualizations if path provided
        if save_path and results:
            self.save_explanations(results, save_path, original_image)
        
        return results
    
    def generate_attention_maps(
        self,
        input_tensor: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate attention maps from intermediate layers
        
        Args:
            input_tensor: Input tensor
            layer_names: Specific layer names to visualize
            
        Returns:
            Dictionary of attention maps
        """
        input_tensor = input_tensor.to(self.device)
        attention_maps = {}
        
        # Hook to capture intermediate activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        hooks = []
        
        # Register hooks
        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Generate attention maps
        for name, activation in activations.items():
            # Take mean across channels for visualization
            attention = torch.mean(activation, dim=1, keepdim=True)
            attention = F.relu(attention)
            
            # Normalize
            attention = attention.squeeze().cpu().numpy()
            attention = attention - np.min(attention)
            if np.max(attention) != 0:
                attention = attention / np.max(attention)
            
            attention_maps[name] = attention
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def generate_feature_importance(
        self,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Generate feature importance using integrated gradients
        
        Args:
            input_tensor: Input tensor
            baseline: Baseline for comparison (if None, use zeros)
            n_steps: Number of integration steps
            
        Returns:
            Feature importance map
        """
        input_tensor = input_tensor.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(self.device)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            target_class = torch.argmax(output, dim=1)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
            
            # Store gradient
            gradients.append(interpolated.grad.clone())
        
        # Calculate integrated gradients
        integrated_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = integrated_gradients * (input_tensor - baseline)
        
        # Convert to importance map
        importance = torch.mean(torch.abs(integrated_gradients), dim=1, keepdim=True)
        importance = importance.squeeze().cpu().detach().numpy()
        
        # Normalize
        importance = importance - np.min(importance)
        if np.max(importance) != 0:
            importance = importance / np.max(importance)
        
        return importance
    
    def save_explanations(
        self,
        explanations: Dict[str, Dict],
        save_path: str,
        original_image: np.ndarray
    ) -> None:
        """Save explanation visualizations"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create a figure with multiple subplots
        n_methods = len(explanations)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
        if n_methods == 1:
            axes = [axes]
        elif (n_methods + 1) // 2 == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot original image
        if n_methods > 0:
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
        
        # Plot explanations
        for idx, (method_name, result) in enumerate(explanations.items()):
            ax_idx = idx + 1 if n_methods > 1 else idx
            
            if ax_idx < len(axes):
                axes[ax_idx].imshow(result['visualization'])
                
                # Add prediction info to title
                pred_class = self.class_names[result['predicted_class']]
                confidence = result['confidence']
                axes[ax_idx].set_title(f'{method_name}\\n{pred_class} ({confidence:.3f})')
                axes[ax_idx].axis('off')
                
                # Save individual explanation
                individual_path = save_path / f'{method_name.lower()}_explanation.png'
                plt.imsave(individual_path, result['visualization'])
        
        # Hide unused subplots
        for idx in range(len(explanations) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'all_explanations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Explanations saved to {save_path}")
    
    def create_explanation_report(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        image_path: str,
        output_dir: str = "explanations"
    ) -> Dict:
        """Create comprehensive explanation report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        report = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist()[0]
        }
        
        # Generate Grad-CAM explanations
        gradcam_results = self.generate_gradcam_explanations(
            input_tensor, original_image,
            save_path=output_path / 'gradcam'
        )
        
        if gradcam_results:
            report['gradcam_generated'] = True
            report['gradcam_methods'] = list(gradcam_results.keys())
        
        # Generate attention maps
        attention_maps = self.generate_attention_maps(input_tensor)
        if attention_maps:
            report['attention_maps_generated'] = True
            report['attention_layers'] = list(attention_maps.keys())
            
            # Save attention maps
            attention_path = output_path / 'attention_maps'
            attention_path.mkdir(exist_ok=True)
            
            for layer_name, attention in attention_maps.items():
                plt.figure(figsize=(8, 6))
                plt.imshow(attention, cmap='hot', interpolation='bilinear')
                plt.colorbar()
                plt.title(f'Attention Map - {layer_name}')
                plt.axis('off')
                
                safe_name = layer_name.replace('.', '_').replace('/', '_')
                plt.savefig(attention_path / f'{safe_name}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Generate feature importance
        try:
            importance = self.generate_feature_importance(input_tensor)
            report['feature_importance_generated'] = True
            
            # Save feature importance
            plt.figure(figsize=(8, 6))
            plt.imshow(importance, cmap='hot', interpolation='bilinear')
            plt.colorbar()
            plt.title('Feature Importance (Integrated Gradients)')
            plt.axis('off')
            plt.savefig(output_path / 'feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.warning(f"Could not generate feature importance: {e}")
            report['feature_importance_generated'] = False
        
        # Save report
        import json
        with open(output_path / 'explanation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Explanation report created: {output_path}")
        
        return report


if __name__ == "__main__":
    # Test explainability module
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from models.architectures import create_model
        from configs.config import get_config
        
        config = get_config()
        
        # Create a dummy model for testing
        model = create_model("resnet50", num_classes=2, pretrained=False)
        
        # Create dummy data
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test explainer
        explainer = AdvancedExplainer(
            model=model,
            class_names=["No Fracture", "Fracture"]
        )
        
        # Generate explanations
        report = explainer.create_explanation_report(
            dummy_input,
            dummy_image,
            "test_image.jpg",
            "test_explanations"
        )
        
        print("Explanation Report:")
        for key, value in report.items():
            print(f"{key}: {value}")
        
        print("✅ Explainability module test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if dependencies are not installed.")