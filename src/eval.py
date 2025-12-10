import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

def get_transforms(model_type: str):
    model_type = model_type.lower()
    if model_type == "dino":
        return transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


def get_val_loader(model_type: str,
                   data_root: str,
                   batch_size: int = 32):
    transform = get_transforms(model_type)

    full_dataset = datasets.ImageFolder(
        root=data_root,
        transform=transform,
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return val_loader, full_dataset.classes



class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class DinoV2Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        import timm  # local import

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
            feats = self.backbone(x)  # [B, D]
        logits = self.classifier(feats)
        return logits


def build_model(model_type: str, num_classes: int, device: torch.device) -> nn.Module:
    model_type = model_type.lower()

    if model_type == "cnn":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_type == "vit":
        model = ViTClassifier(num_classes=num_classes)

    elif model_type == "dino":
        model = DinoV2Classifier(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use one of: cnn, vit, dino.")

    return model.to(device)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            _, preds = torch.max(outputs, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / total
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    acc = accuracy_score(y_true, y_pred)

    return avg_loss, acc, y_true, y_pred


def infer_model_type_from_path(path: str) -> str:
    lower = path.lower()
    if "dino" in lower:
        return "dino"
    if "vit" in lower:
        return "vit"
    if "cnn" in lower:
        return "cnn"
    raise ValueError(
        "Could not infer model type from checkpoint path. "
        "Pass --model-type cnn|vit|dino explicitly."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved .pth model checkpoint on the validation set."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to .pth checkpoint file (e.g. cnn_checkpoints/2.pth, vit_checkpoints/4.pth, dino_checkpoints/6.pth)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "vit", "dino"],
        help="Model type (cnn, vit, dino). If omitted, will try to infer from checkpoint path.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../diabetic_retinopathy_dataset/colored_images",
        help="Path to ImageFolder root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )

    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_type = args.model_type or infer_model_type_from_path(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Model type: {model_type}")

    # Data
    val_loader, class_names = get_val_loader(
        model_type=model_type,
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Val size:", len(val_loader.dataset))

    # Model
    model = build_model(model_type, num_classes=num_classes, device=device)

    # Load checkpoint
    print("\nLoading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", None)
        stored_val_acc = ckpt.get("val_acc", None)
    else:
        # In case you ever saved just state_dict(model)
        model.load_state_dict(ckpt)
        epoch = None
        stored_val_acc = None

    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    if stored_val_acc is not None:
        print(f"Checkpoint stored val_acc: {stored_val_acc:.4f}")

    # Eval
    print("\nEvaluating on validation split...")
    avg_loss, acc, y_true, y_pred = evaluate(model, val_loader, device)

    print("\n=== Evaluation Results ===")
    print(f"Val Loss: {avg_loss:.4f}")
    print(f"Val Acc : {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
