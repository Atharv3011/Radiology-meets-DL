"""
Advanced Dataset and Data Loading for Bone Fracture Detection
Enhanced data pipeline with professional augmentations and efficient loading
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from collections import Counter
import logging
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import get_config


class FractureDataset(Dataset):
    """Enhanced dataset class for fracture detection"""
    
    def __init__(
        self,
        data_dir: str,
        task_type: str = "binary",  # "binary" or "multiclass"
        transform: Optional[transforms.Compose] = None,
        img_size: int = 384,
        use_cache: bool = True
    ):
        """
        Args:
            data_dir: Path to data directory
            task_type: Type of classification task
            transform: Image transformations
            img_size: Image size for resizing
            use_cache: Whether to cache images in memory
        """
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        self.img_size = img_size
        self.use_cache = use_cache
        
        # Load data
        self.image_paths, self.labels, self.class_names = self._load_data()
        
        # Cache for images
        self.image_cache = {} if use_cache else None
        
        logging.info(f"Loaded {len(self.image_paths)} images for {task_type} classification")
        logging.info(f"Classes: {self.class_names}")
        
    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load image paths and labels"""
        image_paths = []
        labels = []
        
        if self.task_type == "binary":
            # For MURA-style binary classification
            class_names = ["No Fracture", "Fracture"]
            
            for body_part in os.listdir(self.data_dir):
                body_path = self.data_dir / body_part
                if not body_path.is_dir():
                    continue
                    
                for patient in os.listdir(body_path):
                    patient_path = body_path / patient
                    if not patient_path.is_dir():
                        continue
                        
                    for study in os.listdir(patient_path):
                        study_path = patient_path / study
                        if not study_path.is_dir():
                            continue
                            
                        # Determine label from study name
                        label = 1 if "positive" in study.lower() else 0
                        
                        for img_name in os.listdir(study_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                img_path = study_path / img_name
                                image_paths.append(str(img_path))
                                labels.append(label)
                                
        else:  # multiclass
            # For fracture type classification
            class_names = []
            class_to_idx = {}
            
            for class_dir in sorted(os.listdir(self.data_dir)):
                class_path = self.data_dir / class_dir
                if not class_path.is_dir():
                    continue
                    
                class_names.append(class_dir)
                class_to_idx[class_dir] = len(class_names) - 1
                
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = class_path / img_name
                        image_paths.append(str(img_path))
                        labels.append(class_to_idx[class_dir])
        
        return image_paths, labels, class_names
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image"""
        # Check cache first
        if self.use_cache and path in self.image_cache:
            return self.image_cache[path]
        
        # Load image
        try:
            # Try with PIL first
            image = Image.open(path).convert('RGB')
            image = np.array(image)
        except Exception:
            # Fallback to OpenCV
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cache if enabled
        if self.use_cache:
            self.image_cache[path] = image
            
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Torchvision transform
                image = Image.fromarray(image)
                image = self.transform(image)
        else:
            # Default transform
            image = Image.fromarray(image)
            default_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = default_transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_weights(self) -> List[float]:
        """Calculate class weights for balanced training"""
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.class_names)
        
        weights = []
        for i in range(num_classes):
            if i in label_counts:
                weight = total_samples / (num_classes * label_counts[i])
            else:
                weight = 1.0
            weights.append(weight)
        
        return weights


def get_advanced_transforms(
    img_size: int = 384,
    is_training: bool = True,
    use_albumentations: bool = True
) -> Union[transforms.Compose, A.Compose]:
    """Get advanced image transformations"""
    
    if use_albumentations:
        if is_training:
            transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=20, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.2
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.2
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.3
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.1
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    else:
        # Torchvision transforms
        if is_training:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.1,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    return transform


def get_data_loaders(
    data_dir: str,
    task_type: str = "binary",
    img_size: int = 384,
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, List[float]]:
    """Create train and validation data loaders"""
    
    # Get transforms
    train_transform = get_advanced_transforms(img_size, is_training=True)
    val_transform = get_advanced_transforms(img_size, is_training=False)
    
    # Create full dataset
    full_dataset = FractureDataset(
        data_dir=data_dir,
        task_type=task_type,
        transform=train_transform,
        img_size=img_size
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation dataset
    val_dataset.dataset.transform = val_transform
    
    # Get class weights
    class_weights = full_dataset.get_class_weights()
    
    # Create samplers for balanced training
    if use_weighted_sampling:
        # Calculate sample weights
        sample_weights = []
        for idx in train_dataset.indices:
            label = full_dataset.labels[idx]
            sample_weights.append(class_weights[label])
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logging.info(f"Created data loaders - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    logging.info(f"Class weights: {class_weights}")
    
    return train_loader, val_loader, class_weights


def create_dataset_csv(data_dir: str, output_path: str) -> None:
    """Create CSV file with dataset information"""
    data_entries = []
    data_dir = Path(data_dir)
    
    for class_dir in os.listdir(data_dir):
        class_path = data_dir / class_dir
        if not class_path.is_dir():
            continue
            
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = class_path / img_name
                
                # Get image info
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                except Exception as e:
                    logging.warning(f"Could not read image {img_path}: {e}")
                    continue
                
                data_entries.append({
                    'image_path': str(img_path),
                    'class_name': class_dir,
                    'filename': img_name,
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'size_mb': img_path.stat().st_size / (1024 * 1024)
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(data_entries)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Created dataset CSV with {len(df)} entries: {output_path}")
    
    # Print statistics
    print(f"Dataset Statistics:")
    print(f"Total images: {len(df)}")
    print(f"Classes: {df['class_name'].nunique()}")
    print(f"Class distribution:")
    print(df['class_name'].value_counts())
    print(f"Average image size: {df['width'].mean():.0f}x{df['height'].mean():.0f}")
    print(f"Total size: {df['size_mb'].sum():.2f} MB")


if __name__ == "__main__":
    # Test dataset loading
    config = get_config()
    
    # Test binary classification
    print("Testing binary classification dataset...")
    try:
        train_loader, val_loader, weights = get_data_loaders(
            data_dir=config.data.train_dir,
            task_type="binary",
            batch_size=4
        )
        
        # Test batch loading
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Batch {batch_idx}: {data.shape}, {target.shape}")
            if batch_idx >= 2:
                break
                
        print("✅ Binary dataset loading successful!")
    except Exception as e:
        print(f"❌ Binary dataset loading failed: {e}")
    
    # Test multiclass classification
    print("\nTesting multiclass classification dataset...")
    try:
        train_loader, val_loader, weights = get_data_loaders(
            data_dir=config.data.classification_dir,
            task_type="multiclass",
            batch_size=4
        )
        
        # Test batch loading
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Batch {batch_idx}: {data.shape}, {target.shape}")
            if batch_idx >= 2:
                break
                
        print("✅ Multiclass dataset loading successful!")
    except Exception as e:
        print(f"❌ Multiclass dataset loading failed: {e}")