from pathlib import Path

# -------------------------------------------------
# Project root
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "plant disease study":
    PROJECT_ROOT = PROJECT_ROOT.parent


# -------------------------------------------------
# Dataset
# -------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "full_split"

# -------------------------------------------------
# Training parameters
# -------------------------------------------------
IMAGE_SIZE = (224, 224)        # SAME as baseline (critical for fair comparison)
BATCH_SIZE = 32                # SAME as baseline
NUM_EPOCHS = 30                # SAME as baseline (no extra epochs yet)
LEARNING_RATE = 1e-4           # Lower LR for more stable training
NUM_CLASSES = 38
RANDOM_SEED = 42               # SAME seed for reproducibility

# -------------------------------------------------
# Output paths
# -------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "experiments" / "cnn" / "results" / "03_all_dataset"
MODEL_SAVE_PATH = RESULTS_DIR / "cnn_03_all_dataset_30epochs_model.pth"

