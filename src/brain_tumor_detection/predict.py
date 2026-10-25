import pickle
from brain_tumor_detection.config import (
    DEVICE,
    FEATURE_COLUMNS,
    MRI_MODEL_PATH,
    SCALER_PATH,
    TABULAR_MODEL_PATH,
)
from brain_tumor_detection.dataset import get_mri_transforms
from brain_tumor_detection.models import MRICNN, TabularMLP
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch
import re
import easyocr
import numpy as np
from PIL import Image
_ocr_reader = None


def load_tabular_artifacts() -> tuple[TabularMLP, StandardScaler]:
  """Loads the trained TabularMLP model and fitted StandardScaler."""
  if not TABULAR_MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError(
        "Tabular model or scaler not found in artifacts/. Run train.py first."
    )

  with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)

  model = TabularMLP().to(DEVICE)
  model.load_state_dict(
      torch.load(TABULAR_MODEL_PATH, map_location=DEVICE, weights_only=True)
  )
  model.eval()
  return model, scaler


def load_mri_model() -> MRICNN:
  """Loads the trained MRICNN model."""
  if not MRI_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MRI model not found at {MRI_MODEL_PATH}. Run train.py first."
    )

  model = MRICNN(pretrained=False).to(DEVICE)
  model.load_state_dict(
      torch.load(MRI_MODEL_PATH, map_location=DEVICE, weights_only=True)
  )
  model.eval()
  return model


def predict_from_features(
    feature_values: dict[str, float] | list[float],
    model: TabularMLP | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[bool, float]:
  """Predicts brain tumor presence from the 13 numerical features.

  Returns: (has_tumor, probability_of_tumor)
  """
  if model is None or scaler is None:
    model, scaler = load_tabular_artifacts()

  if isinstance(feature_values, dict):
    ordered_values = [float(feature_values[col]) for col in FEATURE_COLUMNS]
  else:
    ordered_values = [float(v) for v in feature_values]

  features_array = np.array([ordered_values], dtype=np.float32)
  scaled_features = scaler.transform(features_array)
  tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)

  with torch.no_grad():
    logit = model(tensor_input)
    prob = torch.sigmoid(logit).item()

  return prob >= 0.5, prob


def predict_from_mri(
    image: Image.Image,
    model: MRICNN | None = None,
) -> tuple[bool, float]:
  """Predicts brain tumor presence from a PIL MRI scan image.

  Returns: (has_tumor, probability_of_tumor)
  """
  if model is None:
    model = load_mri_model()

  transform = get_mri_transforms(train=False)
  rgb_image = image.convert("RGB")
  tensor_input = transform(rgb_image).unsqueeze(0).to(DEVICE)

  with torch.no_grad():
    logit = model(tensor_input)
    prob = torch.sigmoid(logit).item()

  return prob >= 0.5, prob



def get_ocr_reader():
  global _ocr_reader
  if _ocr_reader is None:
    _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
  return _ocr_reader
def extract_features_from_report_image(image: Image.Image) -> dict[str, float]:
  """Extracts the 13 feature values from a lab/radiomics report image via OCR."""
  reader = get_ocr_reader()
  img_np = np.array(image.convert("RGB"))
  lines = reader.readtext(img_np, detail=0)
  full_text = "\n".join(lines)
  extracted: dict[str, float] = {}
  for col in FEATURE_COLUMNS:
    # Match feature name followed by optional colon/spaces and a float/scientific number
    pattern = rf"{re.escape(col)}\s*[:=\-]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    match = re.search(pattern, full_text, flags=re.IGNORECASE)
    if match:
      extracted[col] = float(match.group(1))
  return extracted



