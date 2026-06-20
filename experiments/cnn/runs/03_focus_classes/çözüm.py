import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# -------------------------------------------------
# Path setup
# -------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from cnn_03_all_model import CNN_03_All_Dataset
import config


FOCUS_EPOCHS = 8
FOCUS_LR = 3e-5

FOCUS_CLASSES = [
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Tomato___Early_blight",
    "Potato___healthy",
]

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    set_seed(config.RANDOM_SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    results_dir = Path("experiments/cnn/results/03_focus_classes")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = results_dir / "cnn_03_focus_classes_model.pth"

    # -------------------------------------------------
    # Transforms (IDENTICAL to 03 model)
    # -------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------------------------------------------------
    # Datasets
    # -------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=config.DATA_DIR / "val",
        transform=val_transform
    )

    # -------------------------------------------------
    # Focused sampling
    # -------------------------------------------------
    class_to_idx = train_dataset.class_to_idx
    focus_class_indices = [
        class_to_idx[name] for name in FOCUS_CLASSES
    ]

    # -------------------------------------------------
    # Class weights for loss (FIX: keep sampling balanced,
    # let the loss function penalize focus-class mistakes instead)
    # -------------------------------------------------
    class_weights = torch.ones(config.NUM_CLASSES)
    for idx in focus_class_indices:
        class_weights[idx] = 3.0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = CNN_03_All_Dataset(
        num_classes=config.NUM_CLASSES
    ).to(device)

    # Load pretrained 03 model
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device)
    )

    # -------------------------------------------------
    # Freeze all layers except the classifier head,
   
    # -------------------------------------------------
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # -------------------------------------------------
    # Loss & Optimizer
    # -------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FOCUS_LR
    )

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(FOCUS_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss_avg = train_loss / len(train_loader)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        print(
            f"[Focus Train] Epoch [{epoch+1}/{FOCUS_EPOCHS}] "
            f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}"
        )

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------
    torch.save(model.state_dict(), model_save_path)
    print(f"Focused model saved to {model_save_path}")

# -------------------------------------------------
if __name__ == "__main__":
    main()