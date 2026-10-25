from pathlib import Path

import torch

# Base Project Directories
# Resolves to: /home/yahyahabib/Projects/brain-tumor-detection
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Dataset Paths
KAGGLE_DATASET_SLUG = "jakeshbohaju/brain-tumor"
CSV_PATH = DATA_DIR / "Brain Tumor.csv"
IMAGES_DIR = DATA_DIR / "Brain Tumor" / "Brain Tumor"

# Saved Artifact Paths
TABULAR_MODEL_PATH = ARTIFACTS_DIR / "tabular_mlp.pth"
MRI_MODEL_PATH = ARTIFACTS_DIR / "mri_cnn.pth"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

# Tabular Feature Columns (13 features in Brain Tumor.csv)
FEATURE_COLUMNS = [
    "Mean",
    "Variance",
    "Standard Deviation",
    "Entropy",
    "Skewness",
    "Kurtosis",
    "Contrast",
    "Energy",
    "ASM",
    "Homogeneity",
    "Dissimilarity",
    "Correlation",
    "Coarseness",
]

TARGET_COLUMN = "Class"
IMAGE_COLUMN = "Image"

# Image & Model Hyperparameters
IMAGE_SIZE = 224
NUM_FEATURES = len(FEATURE_COLUMNS)  # 13
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS_TABULAR = 30
EPOCHS_MRI = 10
RANDOM_SEED = 42
VAL_SPLIT = 0.2

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")