from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = Path(r"/Users/duhanaydin/cursor/plant disease study/data/processed/full_split")
RESULTS_DIR = Path(r"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/08_transfer_learning/stageA")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -------------------------------------------------
# Model: ImageNet pretrained ResNet18.
# Replace the final fc layer to match our number of classes.
# -------------------------------------------------
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Train for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total



@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Training uses light augmentation; validation uses a clean transform.
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    # =============================================
    # Freeze the backbone; train only the final fc layer.
    # This keeps the ImageNet-learned features intact and
    # only learns the plant-disease classification head.
    # =============================================
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3, weight_decay=1e-4
    )

    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch+1:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), RESULTS_DIR / "model_best.pth")
    print(f"Model saved to {RESULTS_DIR / 'model_best.pth'}")


if __name__ == "__main__":
    main()