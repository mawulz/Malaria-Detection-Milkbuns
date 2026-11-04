# Malaria Cell Classification — EfficientNet-B0 + CBAM

Binary classification of malaria-infected vs. uninfected blood cell images using a hybrid EfficientNet-B0 backbone with CBAM (Convolutional Block Attention Module) attention.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Training pipeline for the proposed model (EfficientNet-B0 + CBAM) |
| `main_baseline.py` | Training pipeline for the baseline model (plain EfficientNet-B0) |
| `model.py` | Proposed model architecture: EfficientNet-B0 with CBAM at stages 5-7 |
| `model_baseline.py` | Baseline model architecture: vanilla EfficientNet-B0 |
| `dataset.py` | Dataset loading, augmentation, and 80/10/10 stratified splitting |
| `evaluate.py` | Metrics calculation (Accuracy, Recall, Precision, F1, AUC-ROC) and model profiling |
| `gradcam.py` | Grad-CAM visualization of false negative predictions |

## Dataset

This project uses the [NIH Malaria Cell Images Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).
The dataset should be placed at `cell_images/` relative to the project root with the following structure:

```
cell_images/
├── Parasitized/   (13,780 images)
└── Uninfected/    (13,780 images)
```

If the dataset is not present, the code will attempt to download it automatically.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train the proposed model (EfficientNet-B0 + CBAM):**
```bash
python main.py
```

**Train the baseline model (plain EfficientNet-B0):**
```bash
python main_baseline.py
```

Both scripts train across 3 seeds (42, 123, 2026) and report mean ± std metrics.

## Key Configuration

- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Input size**: 224 × 224
- **Optimizer**: AdamW (backbone lr=1e-5, head/CBAM lr=1e-4, weight_decay=1e-2)
- **Scheduler**: ReduceLROnPlateau (on F1, patience=5)
- **Early stopping**: patience=10 epochs
- **Max epochs**: 80
- **Threshold tuning**: swept on validation set (0.30–0.60, Precision > 0.95 constraint)

## Outputs

- `training_history.csv` / `training_history_baseline.csv` — per-epoch metrics
- `best_model_seed{seed}.pt` / `checkpoint_baseline_seed_{seed}.pth` — best model weights
- `final_results.txt` / `final_results_baseline.txt` — aggregated results across seeds
- `gradcam_fn_seed{seed}/` / `gradcam_fn_baseline_seed{seed}/` — Grad-CAM visualizations of false negatives
