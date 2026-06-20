from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]# relative path kullandım çünkü bu dosya inference klasöründe


RESULTS_ROOT = PROJECT_ROOT / "experiments" / "cnn" / "results"

MODEL1_DIR = RESULTS_ROOT / "05_model1"
MODEL2_DIR = RESULTS_ROOT / "06_model2"

MODEL1_CKPT = MODEL1_DIR / "model1_baseline_cnn.pth"

IMAGE_SIZE = 224

# Model-2 single-class (OOD/prototype)
SINGLE_PLANTS = {"blueberry", "orange", "raspberry", "soybean", "squash"}

# Threshold choice for single-class OOD
SINGLE_THRESHOLD_KEY = "threshold_p99"  # conservative


DEFAULT_DEVICE = None  #  mps if available else cpu



#cnn
GLOBAL_CNN03_DIR = RESULTS_ROOT / "03_all_dataset"
GLOBAL_CNN03_CKPT = GLOBAL_CNN03_DIR / "cnn_03_all_dataset_30epochs_model.pth"
GLOBAL_CNN03_IDX_TO_CLASS = GLOBAL_CNN03_DIR / "global_cnn_idx_to_class.json"


#cnn + svm
CNN_SVM07_DIR = RESULTS_ROOT / "07_cnn_svm"
CNN_SVM07_EXTRACTOR_CKPT = CNN_SVM07_DIR / "cnn_feature_extractor.pth"
CNN_SVM07_SVM_JOBLIB = CNN_SVM07_DIR / "svm_model.joblib"
CNN_SVM07_IDX_TO_CLASS = CNN_SVM07_DIR / "idx_to_class.json"


# 08_transfer_learning (StageA)
TL08_DIR = RESULTS_ROOT / "08_transfer_learning" / "stageA"
TL08_CKPT = TL08_DIR / "model_best.pth"      
TL08_CLASSES = TL08_DIR / "classes.json"     
TL08_IDX_TO_CLASS = TL08_DIR / "idx_to_class.json"  