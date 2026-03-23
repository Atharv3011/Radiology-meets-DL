import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import random

# ======================
# CONFIG
# ======================
DATA_DIR = "Dataset/train_valid"
IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ======================
# TRANSFORMS
# ======================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# DATASET CLASS
# ======================
class MURA_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for body_part in os.listdir(root_dir):
            body_part_path = os.path.join(root_dir, body_part)
            if not os.path.isdir(body_part_path):
                continue
            for patient in os.listdir(body_part_path):
                patient_path = os.path.join(body_part_path, patient)
                if not os.path.isdir(patient_path):
                    continue
                for study in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study)
                    if not os.path.isdir(study_path):
                        continue
                    label = 1 if "positive" in study.lower() else 0
                    for img_file in os.listdir(study_path):
                        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.samples.append({
                                "path": os.path.join(study_path, img_file),
                                "label": label
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["label"]

# ======================
# LOAD DATA
# ======================
dataset = MURA_Dataset(DATA_DIR, transform=train_transforms)
print(f"🩻 Total images found: {len(dataset)}")

# Split into train / val (80/20)
indices = list(range(len(dataset)))
random.shuffle(indices)
split = int(0.8 * len(dataset))
train_idx, val_idx = indices[:split], indices[split:]

train_data = torch.utils.data.Subset(dataset, train_idx)
val_data = torch.utils.data.Subset(dataset, val_idx)

# ======================
# BALANCED SAMPLER
# ======================
train_labels = [dataset.samples[i]['label'] for i in train_idx]
class_counts = torch.bincount(torch.tensor(train_labels))
print(f"🔢 Class counts (train): {class_counts.tolist()}")

class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(weights=sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ======================
# MODEL
# ======================
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
model = model.to(DEVICE)

# ======================
# LOSS + OPTIMIZER + SCHEDULER
# ======================
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

# ======================
# TRAINING LOOP
# ======================
best_acc = 0.0

if __name__ == '__main__':
    # ======================
    # TRAINING LOOP
    # ======================
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, running_corrects = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for inputs, labels in progress:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_data)
        train_acc = running_corrects.double() / len(train_data)

        # ----- VALIDATION -----
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_data)
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "binary_fracture_mura_balanced.pth")
            print(f"💾 Saved best model (Val Acc: {val_acc*100:.2f}%)")

    print(f"\n✅ Training Complete! Best Val Acc: {best_acc*100:.2f}%")
