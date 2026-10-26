# Brain Tumor Detection System

A dual-mode medical diagnostic web application built with **PyTorch** and **Streamlit** that detects brain tumors from either **raw MRI scan images** or **13 extracted radiomic/statistical features** (via interactive table editing or automated OCR extraction from report images).

---

## Key Features

- **Mode 1 — MRI Scan Classification (CNN):**
  - Upload raw brain MRI scans (`.jpg`, `.jpeg`, `.png`).
  - Powered by a fine-tuned **ResNet-18** Convolutional Neural Network in PyTorch.
  - Outputs binary classification (**Tumor Detected** vs. **No Tumor Detected**) along with prediction confidence and raw tumor probability.
- **Mode 2 — Radiomic Feature Classification (Tabular MLP + OCR):**
  - **Interactive Data Editor:** Edit all 13 first-order statistical and second-order GLCM texture features in a locked-key table, or load random ground-truth samples directly from the dataset.
  - **Automated Report OCR:** Upload an image of a lab/radiomics report and automatically extract the 13 feature values using **EasyOCR** to populate the table and run instant inference.
  - Powered by a custom **Multi-Layer Perceptron (MLP)** with Batch Normalization, Dropout, and `StandardScaler` preprocessing.
- **Automated Dataset Management:**
  - Automatically downloads and structures the [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor) (3,762 MRI scans + extracted feature CSV) on first run via `kagglehub`.

---

## Dataset Overview

This project uses the **[Brain Tumor Dataset by Jakesh Bohaju](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor)** from Kaggle, containing **3,762 samples** with both raw MRI images (`Image1.jpg` – `Image3762.jpg`) and pre-extracted numerical features in `Brain Tumor.csv`:

| Category | Features Included |
| :--- | :--- |
| **First-Order Statistical Features (5)** | `Mean`, `Variance`, `Standard Deviation`, `Skewness`, `Kurtosis` |
| **Second-Order GLCM Texture Features (8)** | `Entropy`, `Contrast`, `Energy`, `ASM` (Angular Second Moment), `Homogeneity`, `Dissimilarity`, `Correlation`, `Coarseness` |
| **Target (`Class`)** | `1` = Tumor, `0` = No Tumor |

---

## Project Structure

```text
brain-tumor-detection/
├── artifacts/                         # Saved model checkpoints & feature scaler
│   ├── mri_cnn.pth                    # Trained ResNet-18 CNN weights
│   ├── tabular_mlp.pth                # Trained Tabular MLP weights
│   └── scaler.pkl                     # Fitted scikit-learn StandardScaler
├── data/                              # Local dataset directory (auto-downloaded)
│   ├── Brain Tumor/
│   │   └── Brain Tumor/               # 3,762 MRI scan images (.jpg)
│   └── Brain Tumor.csv                # Tabular features & class labels
├── src/
│   └── brain_tumor_detection/
│       ├── __init__.py                # Package entry point
│       ├── config.py                  # Paths, feature columns, and hyperparameters
│       ├── dataset.py                 # Kaggle downloader, PyTorch Datasets & transforms
│       ├── models.py                  # TabularMLP and MRICNN (ResNet-18) architectures
│       ├── train.py                   # Training & validation loops for both models
│       ├── predict.py                 # Inference helpers & EasyOCR report parser
│       └── streamlit_app.py           # Streamlit web interface
├── pyproject.toml                     # Project metadata & dependencies (managed by uv)
├── uv.lock                            # Locked dependency versions
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python** `>= 3.13`
- **[uv](https://docs.astral.sh/uv/)** package manager
- *(Optional)* NVIDIA GPU with CUDA for accelerated training and OCR

### 1. Install Dependencies

Clone the repository and sync the environment using `uv`:

```bash
git clone https://github.com/yahyahabib65/brain-tumor-detection.git
cd brain-tumor-detection
uv sync
```

### 2. Train the Models

Run the training pipeline to automatically fetch the dataset (if not already in `data/`) and train both the **Tabular MLP** and **MRI CNN** models:

```bash
uv run python -m brain_tumor_detection.train
```

*(Or if using the script entry point in `pyproject.toml`: `uv run train`)*

Trained artifacts will be saved to `artifacts/`:
- `artifacts/tabular_mlp.pth`
- `artifacts/mri_cnn.pth`
- `artifacts/scaler.pkl`

### 3. Launch the Streamlit App

Start the interactive web application:

```bash
uv run streamlit run src/brain_tumor_detection/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision (ResNet-18)
- **Data Science & Preprocessing:** Pandas, NumPy, Scikit-Learn (`StandardScaler`)
- **Optical Character Recognition (OCR):** EasyOCR
- **Frontend UI:** Streamlit
- **Dataset Integration:** KaggleHub
- **Package & Environment Management:** `uv`

---

## Author

**Yahya Habib** — [yahyahabib65@gmail.com](mailto:yahyahabib65@gmail.com)
