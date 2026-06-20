from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ----------------------------
# CONFIG
# ----------------------------
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Threshold = a quantile of the validation distances.
# p95 is more aggressive (flags more), p99 is more conservative.
THRESHOLD_Q95 = 0.95
THRESHOLD_Q99 = 0.99

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(42)


# ----------------------------
# Pretrained ResNet18 as a feature extractor.
# We drop the classification head so each image maps to a 512-d embedding.
# ----------------------------
def build_feature_extractor():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()   # remove classifier -> output is the 512-d embedding
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embs = []
    for x, _ in loader:
        z = model(x.to(DEVICE))
        z = nn.functional.normalize(z, dim=1)   # unit vectors -> cosine distance
        embs.append(z.cpu())
    return torch.cat(embs, dim=0)


def cosine_distance(emb, prototype):
    # distance = 1 - cosine similarity
    return 1.0 - (emb @ prototype.T).squeeze(1)


def main(plant: str):
    data_dir = Path(f"/Users/duhanaydin/cursor/plant disease study/data/processed/model2_{plant}")
    save_dir = Path(f"/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/06_model2/{plant}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_feature_extractor().to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=transform)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 1) Build the class prototype = mean embedding of all training images.
    #    This is the reference "center" of the only class we have for this plant.
    train_emb = extract_embeddings(model, train_loader)
    prototype = train_emb.mean(dim=0, keepdim=True)
    prototype = nn.functional.normalize(prototype, dim=1)

    # 2) Measure how far each validation image is from the prototype,
    #    then use those distances to pick anomaly thresholds.
    val_emb = extract_embeddings(model, val_loader)
    val_dist = cosine_distance(val_emb, prototype)# ne kadar benziyor?
    thr_p95 = float(torch.quantile(val_dist, THRESHOLD_Q95))
    thr_p99 = float(torch.quantile(val_dist, THRESHOLD_Q99))

    # 3) Evaluate on test: how many images exceed the threshold (flagged as anomalies).
    test_emb = extract_embeddings(model, test_loader)
    test_dist = cosine_distance(test_emb, prototype)
    flag_rate_p95 = float((test_dist > thr_p95).float().mean())
    flag_rate_p99 = float((test_dist > thr_p99).float().mean())

    print(f"Plant: {plant} | class: {train_ds.classes[0]}")
    print(f"Threshold p95: {thr_p95:.4f} | flagged: {flag_rate_p95:.2%}")
    print(f"Threshold p99: {thr_p99:.4f} | flagged: {flag_rate_p99:.2%}")

    #  so weights are reusable.
    torch.save(prototype, save_dir / "prototype.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", required=True)
    args = parser.parse_args()
    main(args.plant)