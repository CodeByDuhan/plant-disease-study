from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "plant disease study":
    PROJECT_ROOT = PROJECT_ROOT.parent



DATA_DIR = PROJECT_ROOT / "data" / "processed" / "full_split"


IMAGE_SIZE = (224, 224)        
BATCH_SIZE = 32                
NUM_EPOCHS = 30                
LEARNING_RATE = 1e-4           
NUM_CLASSES = 38
RANDOM_SEED = 42               


RESULTS_DIR = PROJECT_ROOT / "experiments" / "cnn" / "results" / "03_all_dataset"
MODEL_SAVE_PATH = RESULTS_DIR / "cnn_03_all_dataset_30epochs_model.pth"

