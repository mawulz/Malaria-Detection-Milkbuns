# EfficientNetB0 vs Hybrid EfficientNetB0-CBAM for Malaria Detection 🔬🦟

This repository contains the official implementation of a thesis project benchmarking the standard **EfficientNetB0** architecture against a custom **Hybrid EfficientNetB0-CBAM** (Convolutional Block Attention Module) model for the classification of thin blood smear malaria images.

## Project Overview
The primary goal of this research is to analyze whether integrating spatial and channel-wise attention modules (CBAM) sequentially into the intermediate Mobile Inverted Bottleneck Convolution (MBConv) blocks (Stages 3 through 8) of a pre-trained EfficientNetB0 backbone improves feature extraction and classification metrics on the **NIH Malaria Cell Images Dataset**.

Both models are evaluated fairly utilizing an identical data-splitting distribution (80/10/10 reproducible splits), matching Two-Stage Transfer Learning cycles (Frozen Early Stages + Unfrozen Fine-tuning), and Mixed-Precision (FP16) pipelines.

## Project Structure
* `model_baseline.py` & `main_baseline.py`: Constructs and trains the standard torchvision `EfficientNetB0` baseline architecture.
* `model.py` & `main.py`: Constructs and trains the custom `Hybrid EfficientNetB0-CBAM` attention network.
* `dataset.py`: Handles transparent downloading of the NIH dataset, dataset initialization, validation splitting, and mapping `torchvision.transforms` normalization.
* `evaluate.py`: Standalone mathematical functions that strictly generate robust F1, AUC-ROC, Recall, FP/FN extraction via `Confusion Matrix`, and Parameter/FLOP calculations.
* `profile_baseline.py` & `test_model.py`: Local utility scripts strictly to evaluate MACs/FLOPs, Inference ms timings, and parameter sizing for respective networks based on [Batch, 3, 224, 224] dummy tensors.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cnn-cbam-malaria.git
cd cnn-cbam-malaria
```

### 2. Environment Setup
It is highly recommended to use an isolated Python Virtual Environment (`venv` or `conda`). Once active, install the generic dependencies:
```bash
pip install -r requirements.txt
```
*(Note for GPU users: You may need to install the CUDA-specific version of PyTorch directly depending on your available local NVIDIA driver layout prior to launching training).*

### 3. Data Preparation
The `dataset.py` file controls data hydration implicitly. If the dataset does not exist, the script attempts an automated fetch of the standard NIH ZIP. If the NIH firewall blocks algorithmic queries, the system cleanly errors and prompts explicit extraction of a downloaded mirror into a `/cell_images/` workspace folder.

---

## Execution Guide

### Profiling Network Topologies
To statically evaluate theoretical baseline properties (Parameter count, MegaBytes size, FLOPs, Inference time):
```bash
# Benchmark the standard EfficientNetB0
python profile_baseline.py

# Benchmark the bespoke Hybrid EfficientNetB0-CBAM
python test_model.py
```

### Initiating the Training Loop
To kick off end-to-end multi-seed model compilation, utilizing early stopping and cosine annealing over FP16:

**Run Baseline Model:**
```bash
python main_baseline.py
```
Outputs append chronologically to `training_history_baseline.csv` and averaged conclusions stream to `final_results_baseline.txt`. Model weights will save to `checkpoint_baseline_seed_{X}.pth`.

**Run CBAM Hybrid Model:**
```bash
python main.py
```
Outputs append chronologically to `training_history.csv` and averaged conclusions stream to `final_results.txt`.
