"""
Phase 0: Checkpoint Registry & NIH Test-Set Sanity Checks
==========================================================
Builds a checkpoint registry (SHA256 + file path) and re-evaluates every
checkpoint on the NIH test set at threshold 0.30 to confirm reproducibility.

Outputs (in analysis_outputs/):
  - checkpoint_registry.csv
  - nih_sanity_check.csv

Usage:
    python phase0_sanity.py

Analysis choices:
  - Threshold: 0.30 (fixed, matches training)
  - Dataset split: stratified 80/10/10, random_state=42
  - Inference: fp32 with autocast on CUDA for speed (matches existing eval scripts)
  - Batch size: 32, num_workers: 2
  - Tolerance for metric match: 1e-3 (floating-point tolerance)
"""

import os
import sys
import csv
import hashlib
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 2026, 7, 17]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_outputs')

THRESHOLD = 0.30
BATCH_SIZE = 32
NUM_WORKERS = 2
METRIC_TOLERANCE = 1e-3

# Checkpoint filename patterns
CHECKPOINT_PATTERNS = {
    'hybrid_cbam': 'best_model_seed{seed}.pt',
    'hybrid_eca':  'best_model_ecanet_seed{seed}.pt',
    'baseline':    'checkpoint_baseline_seed_{seed}.pth',
}

# Model builders (imported lazily to avoid circular issues)
MODEL_BUILDERS = {
    'hybrid_cbam': ('model', 'build_model'),
    'hybrid_eca':  ('model_ecanet', 'build_model'),
    'baseline':    ('model_baseline', 'build_model'),
}


def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_checkpoint_registry():
    """Scan for all checkpoints matching the patterns, compute SHA256."""
    rows = []
    for model_type, pattern in CHECKPOINT_PATTERNS.items():
        for seed in SEEDS:
            fname = pattern.format(seed=seed)
            fpath = os.path.join(SCRIPT_DIR, fname)
            if os.path.isfile(fpath):
                sha = sha256_file(fpath)
                rows.append({
                    'model_type': model_type,
                    'seed': seed,
                    'filename': fname,
                    'filepath': fpath,
                    'sha256': sha,
                    'status': 'found',
                })
            else:
                rows.append({
                    'model_type': model_type,
                    'seed': seed,
                    'filename': fname,
                    'filepath': '',
                    'sha256': '',
                    'status': 'MISSING',
                })
    return rows


def get_build_fn(model_type):
    """Dynamically import the model builder."""
    module_name, func_name = MODEL_BUILDERS[model_type]
    sys.path.insert(0, SCRIPT_DIR)
    mod = __import__(module_name)
    return getattr(mod, func_name)


def get_test_loader():
    """Reproduce the exact same test split as the training scripts."""
    sys.path.insert(0, SCRIPT_DIR)
    from dataset import MalariaDataset

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = MalariaDataset(root_dir=DATA_DIR, transform=None)
    total_imgs = len(full_dataset)
    targets = full_dataset.labels.tolist()

    train_idx, temp_idx = train_test_split(
        list(range(total_imgs)), test_size=0.2,
        stratify=targets, random_state=42
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=temp_targets, random_state=42
    )

    test_dataset = MalariaDataset(root_dir=DATA_DIR, transform=val_test_transform)
    test_dataset.image_paths = [full_dataset.image_paths[i] for i in test_idx]
    test_dataset.labels = full_dataset.labels[test_idx]

    print(f"Test set: {len(test_dataset)} cells "
          f"({int((test_dataset.labels == 1).sum())} parasitized, "
          f"{int((test_dataset.labels == 0).sum())} uninfected)")

    loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return loader


def evaluate_checkpoint(model, loader, device, threshold=THRESHOLD):
    """Run inference on the test set and return metrics dict + raw arrays."""
    from evaluate import calculate_metrics

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    metrics = calculate_metrics(all_labels, all_probs, threshold=threshold)
    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 1: Checkpoint Registry ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 0 — Step 1: Checkpoint Registry")
    print("=" * 60)

    registry = build_checkpoint_registry()

    # Print table
    print(f"\n{'Model':<14} {'Seed':<6} {'Status':<8} {'Filename':<40} {'SHA256 (first 16)'}")
    print("-" * 100)
    missing_count = 0
    for r in registry:
        sha_short = r['sha256'][:16] if r['sha256'] else 'N/A'
        flag = " *** MISSING ***" if r['status'] == 'MISSING' else ""
        print(f"{r['model_type']:<14} {r['seed']:<6} {r['status']:<8} "
              f"{r['filename']:<40} {sha_short}{flag}")
        if r['status'] == 'MISSING':
            missing_count += 1

    if missing_count:
        print(f"\n⚠ WARNING: {missing_count} checkpoint(s) MISSING — they will be skipped.")

    # Save registry CSV
    reg_csv = os.path.join(OUTPUT_DIR, 'checkpoint_registry.csv')
    with open(reg_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_type', 'seed', 'filename', 'filepath', 'sha256', 'status'
        ])
        writer.writeheader()
        writer.writerows(registry)
    print(f"\nSaved: {reg_csv}")

    # ── Step 2: NIH Test-Set Sanity Check ───────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 0 — Step 2: NIH Test-Set Re-evaluation")
    print("=" * 60)

    test_loader = get_test_loader()

    sanity_rows = []
    for r in registry:
        if r['status'] == 'MISSING':
            sanity_rows.append({
                'model_type': r['model_type'],
                'seed': r['seed'],
                'sha256': '',
                'Accuracy': '',
                'Recall': '',
                'Precision': '',
                'F1-Score': '',
                'AUC-ROC': '',
                'status': 'SKIPPED (missing checkpoint)',
            })
            continue

        print(f"\n  Evaluating {r['model_type']} seed {r['seed']} ...")
        build_fn = get_build_fn(r['model_type'])
        model = build_fn().to(device)
        state = torch.load(r['filepath'], map_location=device, weights_only=True)
        model.load_state_dict(state)

        metrics = evaluate_checkpoint(model, test_loader, device)

        print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"    Recall:    {metrics['Recall']:.4f}")
        print(f"    Precision: {metrics['Precision']:.4f}")
        print(f"    F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"    AUC-ROC:   {metrics['AUC-ROC']:.4f}")

        sanity_rows.append({
            'model_type': r['model_type'],
            'seed': r['seed'],
            'sha256': r['sha256'],
            'Accuracy': f"{metrics['Accuracy']:.6f}",
            'Recall': f"{metrics['Recall']:.6f}",
            'Precision': f"{metrics['Precision']:.6f}",
            'F1-Score': f"{metrics['F1-Score']:.6f}",
            'AUC-ROC': f"{metrics['AUC-ROC']:.6f}",
            'TN': metrics['TN'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'TP': metrics['TP'],
            'status': 'OK',
        })

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save sanity check CSV
    san_csv = os.path.join(OUTPUT_DIR, 'nih_sanity_check.csv')
    fieldnames = ['model_type', 'seed', 'sha256', 'Accuracy', 'Recall',
                  'Precision', 'F1-Score', 'AUC-ROC', 'TN', 'FP', 'FN', 'TP', 'status']
    with open(san_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sanity_rows)
    print(f"\nSaved: {san_csv}")

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 0 — Summary")
    print("=" * 60)
    ok_count = sum(1 for r in sanity_rows if r['status'] == 'OK')
    skip_count = sum(1 for r in sanity_rows if 'SKIPPED' in str(r['status']))
    print(f"  Evaluated: {ok_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Total:     {len(sanity_rows)}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("  Files: checkpoint_registry.csv, nih_sanity_check.csv")


if __name__ == '__main__':
    main()
