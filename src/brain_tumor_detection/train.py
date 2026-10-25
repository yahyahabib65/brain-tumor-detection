import pickle
from brain_tumor_detection.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEVICE,
    EPOCHS_MRI,
    EPOCHS_TABULAR,
    FEATURE_COLUMNS,
    IMAGE_COLUMN,
    LEARNING_RATE,
    MRI_MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TABULAR_MODEL_PATH,
    TARGET_COLUMN,
    VAL_SPLIT,
)
from brain_tumor_detection.dataset import (
    BrainTumorMRIDataset,
    BrainTumorTabularDataset,
    get_mri_transforms,
    load_dataframe,
)
from brain_tumor_detection.models import MRICNN, TabularMLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
  """Evaluates a binary classifier and returns (avg_loss, accuracy)."""
  model.eval()
  total_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in loader:
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      logits = model(inputs)
      loss = criterion(logits, labels)

      total_loss += loss.item() * inputs.size(0)
      preds = (torch.sigmoid(logits) >= 0.5).float()
      correct += (preds == labels).sum().item()
      total += labels.size(0)

  return total_loss / total, correct / total


def train_tabular_model(train_df, val_df) -> None:
  """Fits scaler, trains TabularMLP, and saves both to artifacts/."""
  print("\n--- Training Tabular MLP Model ---")
  ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
  X_val = scaler.transform(val_df[FEATURE_COLUMNS].values)

  y_train = train_df[TARGET_COLUMN].values
  y_val = val_df[TARGET_COLUMN].values

  # Save fitted scaler for inference
  with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
  print(f"Saved scaler to {SCALER_PATH}")

  train_dataset = BrainTumorTabularDataset(X_train, y_train)
  val_dataset = BrainTumorTabularDataset(X_val, y_val)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  model = TabularMLP().to(DEVICE)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  best_val_acc = 0.0
  for epoch in range(1, EPOCHS_TABULAR + 1):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      optimizer.zero_grad()
      logits = model(inputs)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_dataset)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), TABULAR_MODEL_PATH)

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS_TABULAR:
      print(
          f"Epoch [{epoch:02d}/{EPOCHS_TABULAR:02d}] "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc:"
          f" {val_acc:.4%}"
      )

  print(
      f"Best Tabular Val Accuracy: {best_val_acc:.4%} (Saved to"
      f" {TABULAR_MODEL_PATH})"
  )


def train_mri_model(train_df, val_df) -> None:
  """Trains MRICNN on MRI scan images and saves best weights to artifacts/."""
  print(f"\n--- Training MRI CNN Model on {DEVICE} ---")
  ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

  train_dataset = BrainTumorMRIDataset(
      train_df[IMAGE_COLUMN].values,
      train_df[TARGET_COLUMN].values,
      transform=get_mri_transforms(train=True),
  )
  val_dataset = BrainTumorMRIDataset(
      val_df[IMAGE_COLUMN].values,
      val_df[TARGET_COLUMN].values,
      transform=get_mri_transforms(train=False),
  )

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  model = MRICNN(pretrained=True).to(DEVICE)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)

  best_val_acc = 0.0
  for epoch in range(1, EPOCHS_MRI + 1):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
      optimizer.zero_grad()
      logits = model(inputs)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_dataset)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), MRI_MODEL_PATH)

    print(
        f"Epoch [{epoch:02d}/{EPOCHS_MRI:02d}] "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc:"
        f" {val_acc:.4%}"
    )

  print(
      f"Best MRI Val Accuracy: {best_val_acc:.4%} (Saved to {MRI_MODEL_PATH})"
  )


def main() -> None:
  torch.manual_seed(RANDOM_SEED)
  df = load_dataframe()

  train_df, val_df = train_test_split(
      df,
      test_size=VAL_SPLIT,
      random_state=RANDOM_SEED,
      stratify=df[TARGET_COLUMN],
  )

  train_tabular_model(train_df, val_df)
  train_mri_model(train_df, val_df)


if __name__ == "__main__":
  main()