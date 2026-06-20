from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_03_all_model import CNN_03_All_Dataset


# -------------------------------------------------
# Paths
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = REPO_ROOT / "experiments" / "cnn" / "results" / "07_cnn_svm"
CNN_CKPT = (
    REPO_ROOT / "experiments" / "cnn" / "results"
    / "03_all_dataset" / "cnn_03_all_dataset_30epochs_model.pth"
)
DATA_SPLIT = REPO_ROOT / "data" / "processed" / "full_split"



# (son sınıflandırma katmanını atıp 128-boyutlu embedding'i alıyoruz)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.conv1 = base.conv1
        self.conv2 = base.conv2
        self.conv3 = base.conv3
        self.flatten = base.classifier[0]   # Flatten
        self.fc1 = base.classifier[1]       # Linear(... -> 128)
        self.relu = base.classifier[2]      # ReLU  -> embedding burası

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.relu(x)                 # 128-boyutlu embedding



@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    X, y = [], []
    for images, labels in loader:
        emb = model(images.to(device)).cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # L2 normalize
        X.append(emb)
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = datasets.ImageFolder(DATA_SPLIT / "train", transform=transform)
    test_ds = datasets.ImageFolder(DATA_SPLIT / "test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Eğitilmiş CNN ağırlıklarını yükle, eval moduna al (BatchNorm sabit)
    base = CNN_03_All_Dataset(num_classes=len(train_ds.classes))
    base.load_state_dict(torch.load(CNN_CKPT, map_location=device))
    base.eval()

    feat_model = CNNFeatureExtractor(base).to(device)

    # CNN ile embedding çıkar
    X_train, y_train = extract_embeddings(feat_model, train_loader, device)
    X_test, y_test = extract_embeddings(feat_model, test_loader, device)

    # Bu embedding'ler üzerinde SVM eğit
    svm = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced")
    svm.fit(X_train, y_train)

    
    pred = svm.predict(X_test)
    print("Accuracy    :", accuracy_score(y_test, pred))
    print("Macro Recall:", recall_score(y_test, pred, average="macro"))
    print("Macro F1    :", f1_score(y_test, pred, average="macro"))


if __name__ == "__main__":
    main()