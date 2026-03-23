import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from collections import Counter

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "Dataset/train_valid"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
BASE_LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Using device: {DEVICE}")

# =========================================================
# CUSTOM DATASET FOR MURA STRUCTURE
# =========================================================
class MURADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []  # 1 = fracture, 0 = no fracture
        self.transform = transform

        for body_part in os.listdir(root_dir):
            body_path = os.path.join(root_dir, body_part)
            if not os.path.isdir(body_path):
                continue

            for patient in os.listdir(body_path):
                patient_path = os.path.join(body_path, patient)
                if not os.path.isdir(patient_path):
                    continue

                for study in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study)
                    if not os.path.isdir(study_path):
                        continue

                    label = 1 if "positive" in study.lower() else 0

                    for img_name in os.listdir(study_path):
                        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.image_paths.append(os.path.join(study_path, img_name))
                            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# =========================================================
# AUGMENTATIONS
# =========================================================
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

# =========================================================
# DATASET SPLIT
# =========================================================
dataset = MURADataset(DATA_DIR, transform=train_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

# use validation transforms for val_ds
val_ds.dataset.transform = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================================================
# CLASS WEIGHTS (to fix imbalance)
# =========================================================
label_counts = Counter(dataset.labels)
print(f"🩻 Dataset balance: {label_counts}")
fracture_weight = label_counts[0] / label_counts[1] if label_counts[1] != 0 else 1.0
weights = torch.tensor([1.0, fracture_weight]).to(DEVICE)
print(f"⚖️ Using class weights: {weights.tolist()}")

# =========================================================
# MODEL (ResNet50 + deeper head)
# =========================================================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False

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
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.fc.parameters(), lr=BASE_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
scaler = torch.amp.GradScaler(device=DEVICE)

# =========================================================
# TRAINING LOOP
# =========================================================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)

    # ---------------- Validation ----------------
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    scheduler.step(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "binary_fracture_mura_advanced.pth")
        print(f"💾 Best model saved (Val Acc: {best_val_acc:.2f}%)")

print("✅ Training complete!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
