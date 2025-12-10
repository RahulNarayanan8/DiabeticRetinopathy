import torch

import os
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
from torch.optim import Adam
import timm

os.makedirs("src/dino_checkpoints", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(
    root="../diabetic_retinopathy_dataset/colored_images",
    transform=transform,
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(full_dataset.classes)

print("Classes:", full_dataset.classes)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Using device:", device)

class DinoV2Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = self.backbone.num_features
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

model = DinoV2Classifier(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Train", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Val  ", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

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
        ckpt_path = f"dino_checkpoints/{epoch + 1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
