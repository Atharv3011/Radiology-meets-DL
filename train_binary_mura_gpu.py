import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_DIR = "Dataset/train_valid"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

# =========================
# DEVICE SELECTION
# =========================
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Using Apple Silicon (MPS)")
elif torch.backends.directml.is_available():
    DEVICE = "directml"
    print("✅ Using GPU (DirectML)")
else:
    DEVICE = "cpu"
    print("⚠️ GPU not found, using CPU")


# =========================
# Custom Dataset for MURA
# =========================
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


# =========================
# MAIN TRAINING FUNCTION
# =========================
def train():
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = MURADataset(DATA_DIR, transform=train_transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # ✅ Use num_workers=0 on Windows to avoid spawn errors
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device=DEVICE)

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

        acc = 100 * correct / total
        print(f"Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "binary_fracture_mura_gpu.pth")
    print("✅ Binary MURA model trained and saved with GPU support!")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    train()
