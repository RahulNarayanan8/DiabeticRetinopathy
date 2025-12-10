import numpy as np
import torch

import os
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("rfc_checkpoints", exist_ok=True)


transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # dataset is already 224x224
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

backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print("Loaded pretrained ResNet-50 as feature extractor.")

in_features = backbone.fc.in_features
backbone.fc = nn.Identity()

backbone = backbone.to(device)
backbone.eval()


@torch.no_grad()
def extract_features(model, loader, desc="Extracting"):
    model.eval()
    all_feats = []
    all_labels = []

    for images, labels in tqdm(loader, desc=desc, unit="batch"):
        images = images.to(device)
        feats = model(images)

        feats = feats.view(feats.size(0), -1)

        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


print("Extracting train features...")
X_train, y_train = extract_features(backbone, train_loader, desc="Train features")
print("Extracting val features...")
X_val, y_val = extract_features(backbone, val_loader, desc="Val features")

print(f"Train features shape: {X_train.shape}")
print(f"Val   features shape: {X_val.shape}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

print("Fitting RandomForest on extracted features...")
rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_val_pred   = rf.predict(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val,   y_val_pred)

print("=== RandomForest on ResNet-50 features ===")
print(f"  Train Acc: {train_acc:.4f}")
print(f"  Val   Acc: {val_acc:.4f}")

print("\nValidation classification report:")
print(classification_report(y_val, y_val_pred, digits=4))

print("Validation confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))
