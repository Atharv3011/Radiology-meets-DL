"""
Comprehensive Evaluation Metrics for Bone Fracture Detection
Professional metrics calculation with medical imaging focus
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging


class MetricsCalculator:
    """Comprehensive metrics calculator for medical image classification"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names or []
        
    def calculate_metrics(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor],
        y_pred_proba: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if y_pred_proba is not None:
            y_pred_proba = self._to_numpy(y_pred_proba)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            metrics[f'precision_{class_name}'] = p
            metrics[f'recall_{class_name}'] = r
            metrics[f'f1_{class_name}'] = f
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Medical-specific metrics
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics.update(self._calculate_binary_metrics(y_true, y_pred, y_pred_proba))
        else:  # Multi-class
            metrics.update(self._calculate_multiclass_metrics(y_true, y_pred, y_pred_proba))
        
        return metrics
    
    def _calculate_binary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate binary classification specific metrics"""
        metrics = {}
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Sensitivity (Recall) and Specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Positive and Negative Predictive Values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Same as precision
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Likelihood ratios
        metrics['lr_positive'] = metrics['sensitivity'] / (1 - metrics['specificity']) if metrics['specificity'] != 1.0 else float('inf')
        metrics['lr_negative'] = (1 - metrics['sensitivity']) / metrics['specificity'] if metrics['specificity'] != 0.0 else float('inf')
        
        # Diagnostic odds ratio
        if metrics['lr_negative'] != 0:
            metrics['diagnostic_odds_ratio'] = metrics['lr_positive'] / metrics['lr_negative']
        else:
            metrics['diagnostic_odds_ratio'] = float('inf')
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denominator != 0:
            metrics['mcc'] = (tp * tn - fp * fn) / mcc_denominator
        else:
            metrics['mcc'] = 0.0
        
        # ROC AUC
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]  # Take positive class probability
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        
        return metrics
    
    def _calculate_multiclass_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate multi-class classification specific metrics"""
        metrics = {}
        
        # Multi-class AUC metrics
        if y_pred_proba is not None:
            n_classes = len(np.unique(y_true))
            
            # One-vs-Rest AUC
            try:
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                if n_classes == 2:
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                
                # Macro-average AUC
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                
                # Weighted-average AUC
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
                
            except Exception as e:
                logging.warning(f"Could not calculate multi-class AUC: {e}")
        
        # Class balance metrics
        class_counts = np.bincount(y_true)
        metrics['class_balance_ratio'] = np.min(class_counts) / np.max(class_counts)
        
        return metrics
    
    def _to_numpy(self, data: Union[List, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert data to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        else:
            return data
    
    def generate_classification_report(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor],
        save_path: Optional[str] = None
    ) -> str:
        """Generate detailed classification report"""
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        target_names = self.class_names if self.class_names else None
        
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logging.info(f"Classification report saved to {save_path}")
        
        return report
    
    def plot_confusion_matrix(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor],
        save_path: Optional[str] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot confusion matrix"""
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else 'auto',
            yticklabels=self.class_names if self.class_names else 'auto'
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred_proba: Union[List, np.ndarray, torch.Tensor],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot ROC curve"""
        y_true = self._to_numpy(y_true)
        y_pred_proba = self._to_numpy(y_pred_proba)
        
        plt.figure(figsize=figsize)
        
        if len(np.unique(y_true)) == 2:  # Binary classification
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
        else:  # Multi-class
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = self.class_names[i] if i < len(self.class_names) else f'Class {i}'
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"ROC curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_precision_recall_curve(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred_proba: Union[List, np.ndarray, torch.Tensor],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot Precision-Recall curve"""
        y_true = self._to_numpy(y_true)
        y_pred_proba = self._to_numpy(y_pred_proba)
        
        plt.figure(figsize=figsize)
        
        if len(np.unique(y_true)) == 2:  # Binary classification
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            
            # Baseline
            baseline = np.sum(y_true) / len(y_true)
            plt.axhline(y=baseline, color='navy', linestyle='--', 
                       label=f'Baseline (AP = {baseline:.3f})')
            
        else:  # Multi-class
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                pr_auc = auc(recall, precision)
                
                class_name = self.class_names[i] if i < len(self.class_names) else f'Class {i}'
                plt.plot(recall, precision, lw=2,
                        label=f'{class_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Precision-Recall curve saved to {save_path}")
        
        return plt.gcf()
    
    def save_metrics_to_json(
        self,
        metrics: Dict,
        save_path: str
    ) -> None:
        """Save metrics to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logging.info(f"Metrics saved to {save_path}")
    
    def create_evaluation_report(
        self,
        y_true: Union[List, np.ndarray, torch.Tensor],
        y_pred: Union[List, np.ndarray, torch.Tensor],
        y_pred_proba: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        output_dir: str = "evaluation_results",
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Create comprehensive evaluation report with plots and metrics"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate classification report
        report = self.generate_classification_report(
            y_true, y_pred,
            save_path=output_path / f"{model_name}_classification_report.txt"
        )
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=output_path / f"{model_name}_confusion_matrix.png"
        )
        
        # Plot normalized confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=output_path / f"{model_name}_confusion_matrix_normalized.png",
            normalize=True
        )
        
        # Plot ROC curve if probabilities available
        if y_pred_proba is not None:
            self.plot_roc_curve(
                y_true, y_pred_proba,
                save_path=output_path / f"{model_name}_roc_curve.png"
            )
            
            self.plot_precision_recall_curve(
                y_true, y_pred_proba,
                save_path=output_path / f"{model_name}_pr_curve.png"
            )
        
        # Save metrics to JSON
        self.save_metrics_to_json(
            metrics,
            save_path=output_path / f"{model_name}_metrics.json"
        )
        
        logging.info(f"Evaluation report created in {output_path}")
        
        return metrics


if __name__ == "__main__":
    # Test the metrics calculator
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    n_classes = 3
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    
    # Add some noise to predictions
    noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.randint(0, n_classes, len(noise_indices))
    
    # Generate dummy probabilities
    y_pred_proba = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    
    # Test metrics calculator
    class_names = ["No Fracture", "Simple Fracture", "Complex Fracture"]
    calculator = MetricsCalculator(class_names=class_names)
    
    # Calculate metrics
    metrics = calculator.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print("Calculated Metrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")
    
    # Create evaluation report
    calculator.create_evaluation_report(
        y_true, y_pred, y_pred_proba,
        output_dir="test_evaluation",
        model_name="test_model"
    )
    
    print("✅ Metrics calculator test completed!")