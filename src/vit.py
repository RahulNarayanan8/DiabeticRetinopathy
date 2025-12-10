
import os
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

os.makedirs("src/vit_checkpoints", exist_ok=True)

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root="../diabetic_retinopathy_dataset/colored_images", transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load a Vision Transformer ----
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
print("Loaded pretrained vision transformer")

# Replace classifier head for your dataset (e.g., num_classes = 5)
num_classes = 5
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

model = model.to(device)

# ---- Loss & Optimizer ----
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Wrap the loader with tqdm
    for images, labels in tqdm(loader, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Optional: update tqdm postfix with dynamic info
        tqdm.write(f"Batch loss: {loss.item():.4f}, Acc: {correct/total:.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if (epoch + 1) % 2 == 0:
        ckpt_path = f"vit_checkpoints/{epoch + 1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

