import shutil
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from brain_tumor_detection.config import (
  CSV_PATH,
  DATA_DIR,
  FEATURE_COLUMNS,
  IMAGE_SIZE,
  IMAGES_DIR,
  KAGGLE_DATASET_SLUG,
  TARGET_COLUMN,
)


def ensure_data_downloaded() -> Path:
  """Downloads the dataset via kagglehub into DATA_DIR if not already present."""
  if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
    print(f"Downloading {KAGGLE_DATASET_SLUG} from Kaggle...")
    cached_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cached_path, DATA_DIR, dirs_exist_ok=True)
    print(f"Dataset copied to {DATA_DIR}")
  return DATA_DIR


def load_dataframe(csv_path: Path = CSV_PATH) -> pd.DataFrame:
  """Loads Brain Tumor.csv and drops any rows with missing values."""
  ensure_data_downloaded()
  df = pd.read_csv(csv_path)
  df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(
      drop=True
  )
  return df


class BrainTumorTabularDataset(Dataset):
  """PyTorch Dataset for the 13 extracted numerical MRI features."""

  def __init__(self, features: np.ndarray, labels: np.ndarray):
    self.features = torch.tensor(features, dtype=torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.float32)

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.features[index], self.labels[index]


class BrainTumorMRIDataset(Dataset):
  """PyTorch Dataset for raw MRI scan images."""

  def __init__(
      self,
      image_names: list[str] | np.ndarray,
      labels: list[int] | np.ndarray,
      img_dir: Path = IMAGES_DIR,
      transform: transforms.Compose | None = None,
  ):
    self.image_names = list(image_names)
    self.labels = torch.tensor(labels, dtype=torch.float32)
    self.img_dir = Path(img_dir)
    self.transform = transform or get_mri_transforms(train=False)

  def __len__(self) -> int:
    return len(self.labels)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    img_name = str(self.image_names[index])
    if not img_name.lower().endswith(".jpg"):
      img_name = f"{img_name}.jpg"

    img_path = self.img_dir / img_name
    pil_image = Image.open(img_path).convert("RGB")

    image_tensor: torch.Tensor = self.transform(pil_image)

    return image_tensor, self.labels[index]


def get_mri_transforms(
    image_size: int = IMAGE_SIZE, train: bool = False
) -> transforms.Compose:
  """Returns torchvision transforms for MRI images."""
  if train:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
  return transforms.Compose([
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])