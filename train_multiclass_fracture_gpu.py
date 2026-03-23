import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# =======================================================
# CONFIG
# =======================================================
DATA_DIR = "Dataset/classification"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4

# =======================================================
# DEVICE
# =======================================================
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Using Apple Silicon (MPS)")
elif hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
    DEVICE = "directml"
    print("✅ Using GPU (DirectML)")
else:
    DEVICE = "cpu"
    print("⚠️ Using CPU")

# =======================================================
# DATA TRANSFORMS
# =======================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

# =======================================================
# DATASET + SPLIT
# =======================================================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
num_classes = len(full_dataset.classes)
print(f"🩻 Found {num_classes} fracture types: {full_dataset.classes}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Use val_transforms for validation set
val_ds.dataset.transform = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =======================================================
# MODEL (ResNet50)
# =======================================================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scaler = torch.amp.GradScaler(device=DEVICE)

# =======================================================
# TRAINING LOOP
# =======================================================
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

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
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# =======================================================
# SAVE MODEL
# =======================================================
torch.save(model.state_dict(), "multiclass_fracture_gpu.pth")
print("✅ Multi-class fracture model trained & saved successfully!")
