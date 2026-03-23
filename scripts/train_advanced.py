"""
Advanced Training Script for Bone Fracture Detection
Features: EfficientNet, ensemble training, advanced augmentation, and comprehensive logging
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.architectures import create_model, create_ensemble_model
from data.advanced_dataset import FractureDataset, get_data_loaders
from data.augmentations import get_advanced_transforms
from configs.config import get_config
from evaluation.metrics import MetricsCalculator


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.fill_(self.smoothing / (self.num_classes - 1))
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -(targets_one_hot * log_probs).sum(dim=1).mean()
        return loss


class AdvancedTrainer:
    """Advanced trainer with comprehensive features"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.setup_logging()
        
        # Initialize models, data, and training components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def _setup_device(self):
        \"\"\"Setup computation device\"\"\"
        if self.config.device == \"auto\":
            if torch.cuda.is_available():
                device = \"cuda\"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = \"mps\"
            else:
                device = \"cpu\"
        else:
            device = self.config.device
            
        logging.info(f\"Using device: {device}\")
        return device
    
    def setup_logging(self):
        \"\"\"Setup logging configuration\"\"\"
        log_dir = Path(\"logs\")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Setup wandb if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__,
                name=f\"{self.config.model.backbone}_{self.config.version}\"
            )
    
    def setup_model(self, num_classes: int, task_type: str = \"binary\"):
        \"\"\"Setup model architecture\"\"\"
        logging.info(f\"Setting up {self.config.model.backbone} model for {task_type} classification\")
        
        if self.config.model.use_ensemble:
            self.model = create_ensemble_model(
                self.config.model.ensemble_models,
                num_classes=num_classes,
                pretrained=self.config.model.pretrained,
                freeze_backbone=self.config.model.freeze_backbone,
                dropout_rate=self.config.model.dropout_rate
            )
        else:
            self.model = create_model(
                self.config.model.backbone,
                num_classes=num_classes,
                pretrained=self.config.model.pretrained,
                freeze_backbone=self.config.model.freeze_backbone,
                dropout_rate=self.config.model.dropout_rate
            )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logging.info(f\"Total parameters: {total_params:,}\")
        logging.info(f\"Trainable parameters: {trainable_params:,}\")
        
        return self.model
    
    def setup_data(self, data_dir: str, task_type: str = \"binary\"):
        \"\"\"Setup data loaders\"\"\"
        logging.info(f\"Setting up data loaders for {task_type} classification\")
        
        self.train_loader, self.val_loader, class_weights = get_data_loaders(
            data_dir=data_dir,
            task_type=task_type,
            img_size=self.config.model.img_size,
            batch_size=self.config.model.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        logging.info(f\"Train samples: {len(self.train_loader.dataset)}\")
        logging.info(f\"Validation samples: {len(self.val_loader.dataset)}\")
        
        return class_weights
    
    def setup_training(self, class_weights=None):
        \"\"\"Setup optimizer, scheduler, and loss function\"\"\"
        # Optimizer
        if self.config.model.freeze_backbone:
            # Only train classifier
            if hasattr(self.model, 'classifier'):
                params = self.model.classifier.parameters()
            else:
                params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()
        
        self.optimizer = optim.AdamW(
            params,
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # Scheduler
        if self.config.model.scheduler == \"cosine\":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.model.epochs,
                eta_min=self.config.model.learning_rate * 0.01
            )
        elif self.config.model.scheduler == \"onecycle\":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.model.learning_rate,
                epochs=self.config.model.epochs,
                steps_per_epoch=len(self.train_loader)
            )
        
        # Loss function
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = FocalLoss(alpha=1, gamma=2)
        
        # Mixed precision scaler
        if self.config.model.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logging.info(f\"Optimizer: {type(self.optimizer).__name__}\")
        logging.info(f\"Scheduler: {type(self.scheduler).__name__}\")
        logging.info(f\"Loss function: {type(self.criterion).__name__}\")
    
    def train_epoch(self):
        \"\"\"Train for one epoch\"\"\"
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f\"Epoch {self.current_epoch+1}/{self.config.model.epochs}\")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.model.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.model.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.model.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.config.model.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.model.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update scheduler for OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        \"\"\"Validate for one epoch\"\"\"
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=\"Validation\"):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config.model.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            all_targets, all_preds
        )
        
        return avg_loss, metrics
    
    def save_checkpoint(self, metrics, is_best=False):
        \"\"\"Save model checkpoint\"\"\"
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'config': self.config.__dict__,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_dir = Path(\"models/checkpoints\")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        latest_path = checkpoint_dir / \"latest_checkpoint.pth\"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / \"best_checkpoint.pth\"
            torch.save(checkpoint, best_path)
            logging.info(f\"New best model saved with validation accuracy: {metrics['accuracy']:.4f}\")
    
    def train(self, data_dir: str, task_type: str = \"binary\", num_classes: int = 2):
        \"\"\"Main training loop\"\"\"
        logging.info(f\"Starting training for {task_type} classification\")
        
        # Setup
        class_weights = self.setup_data(data_dir, task_type)
        self.setup_model(num_classes, task_type)
        self.setup_training(class_weights)
        
        # Training loop
        for epoch in range(self.config.model.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # Update scheduler (except OneCycleLR which updates per step)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['f1_score']
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1_score'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Print epoch results
            logging.info(
                f\"Epoch {epoch+1}/{self.config.model.epochs} | \"
                f\"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | \"
                f\"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | \"
                f\"Val F1: {val_metrics['f1_score']:.4f}\"
            )
        
        logging.info(f\"Training completed! Best validation accuracy: {self.best_val_acc:.4f}\")
        return self.best_val_acc, self.best_val_f1


def main():
    \"\"\"Main training function\"\"\"
    config = get_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.backends.cudnn.deterministic = config.deterministic
        torch.backends.cudnn.benchmark = config.benchmark
    
    # Initialize trainer
    trainer = AdvancedTrainer(config)
    
    # Train binary model
    logging.info(\"=\" * 50)
    logging.info(\"TRAINING BINARY CLASSIFICATION MODEL\")
    logging.info(\"=\" * 50)
    
    binary_acc, binary_f1 = trainer.train(
        data_dir=config.data.train_dir,
        task_type=\"binary\",
        num_classes=2
    )
    
    # Train multiclass model
    logging.info(\"=\" * 50)
    logging.info(\"TRAINING MULTICLASS CLASSIFICATION MODEL\")
    logging.info(\"=\" * 50)
    
    trainer_multi = AdvancedTrainer(config)
    multi_acc, multi_f1 = trainer_multi.train(
        data_dir=config.data.classification_dir,
        task_type=\"multiclass\",
        num_classes=len(config.model.fracture_types)
    )
    
    # Final results
    logging.info(\"=\" * 50)
    logging.info(\"TRAINING SUMMARY\")
    logging.info(\"=\" * 50)
    logging.info(f\"Binary Classification - Accuracy: {binary_acc:.4f}, F1: {binary_f1:.4f}\")
    logging.info(f\"Multiclass Classification - Accuracy: {multi_acc:.4f}, F1: {multi_f1:.4f}\")
    
    if config.use_wandb:
        wandb.finish()


if __name__ == \"__main__\":
    main()