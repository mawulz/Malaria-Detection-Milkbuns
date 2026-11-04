"""
Phase 2: BBBC041 Seed-Collapse Diagnostics
============================================
Comprehensive diagnostics for the BBBC041 zero-shot transfer collapse,
particularly investigating hybrid seed 42.

Analysis choices (cite in methods):
  - Collapse rule: BBBC041 AUC-ROC < 0.90 (pre-specified, fixed)
  - BBBC041 binary mapping: ring/trophozoite/schizont/gametocyte = Parasitized,
    red blood cell = Uninfected, 'difficult' skipped
  - Probability histograms: 50 bins in [0,1]
  - FP export: top 50 highest-probability uninfected crops per seed
  - CBAM/ECA gate statistics: 1,000 random cells from NIH test, 1,000 from BBBC041 test
  - Gate saturation threshold: > 0.95 or < 0.05
  - BatchNorm re-estimation: momentum=None (cumulative), 2,000 UNLABELLED crops
    from BBBC041 TRAINING split, forward-pass only (no grad, no dropout, no labels)
  - Control BN re-estimation: 2,000 NIH training images
  - Threshold: 0.30 throughout (ORACLE threshold only labelled as diagnostic)
  - RNG seed: 12345 for all random sampling
  - Image grids: 5×10 layout for 50 crops, 300 dpi
  - All inference: autocast on CUDA for speed (matching existing pipeline)

Outputs (in analysis_outputs/):
  - bbbc041_diagnostics_per_seed.csv
  - prob_histograms_*.png
  - fp_grid_*.png
  - gate_statistics.csv, gate_comparison.png
  - bn_reestimation.csv, bn_reestimation.png
  - Paper tables appended to paper_tables.md

Usage:
    python bbbc041_diagnostics.py

RNG seeds used: 12345
"""

import os
import sys
import csv
import copy
import json
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy import stats
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 2026, 7, 17]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_outputs')

THRESHOLD = 0.30
BATCH_SIZE = 32
NUM_WORKERS = 2
RANDOM_SEED = 12345
COLLAPSE_AUC_THRESHOLD = 0.90
GATE_SAMPLE_SIZE = 1000
BN_REESTIMATION_SIZE = 2000
HIST_BINS = 50
FP_EXPORT_COUNT = 50
GATE_SATURATION_HIGH = 0.95
GATE_SATURATION_LOW = 0.05

CHECKPOINT_PATTERNS = {
    'hybrid_cbam': 'best_model_seed{seed}.pt',
    'hybrid_eca':  'best_model_ecanet_seed{seed}.pt',
    'baseline':    'checkpoint_baseline_seed_{seed}.pth',
}

MODEL_BUILDERS = {
    'hybrid_cbam': ('model', 'build_model'),
    'hybrid_eca':  ('model_ecanet', 'build_model'),
    'baseline':    ('model_baseline', 'build_model'),
}


def get_build_fn(model_type):
    sys.path.insert(0, SCRIPT_DIR)
    module_map = {
        'hybrid_cbam': 'model',
        'hybrid_eca':  'model_ecanet',
        'baseline':    'model_baseline',
    }
    mod = __import__(module_map[model_type])
    return mod.build_model


def get_bbbc041_test_data():
    """Parse BBBC041 test annotations and create dataset/loader."""
    sys.path.insert(0, SCRIPT_DIR)
    from bbbc041_eval import (parse_bbbc041_annotations, BBBC041_DIR,
                               BBBC041Dataset, create_bbbc041_loader)

    cells, class_counts = parse_bbbc041_annotations(BBBC041_DIR, subset='test')
    # Filter: only 'red blood cell' for uninfected (match prompt spec)
    # The existing code already maps correctly, but leukocyte is also Uninfected
    # In the test set there are no leukocytes, so no filtering needed
    loader = create_bbbc041_loader(cells)
    return cells, loader


def get_bbbc041_training_data():
    """Parse BBBC041 TRAINING annotations for BN re-estimation."""
    sys.path.insert(0, SCRIPT_DIR)
    from bbbc041_eval import (parse_bbbc041_annotations, BBBC041_DIR,
                               BBBC041Dataset)

    cells, _ = parse_bbbc041_annotations(BBBC041_DIR, subset='train')
    return cells


def get_nih_test_loader():
    """Get NIH test loader (same split as training)."""
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

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader, test_dataset


def get_nih_train_loader(n_samples=BN_REESTIMATION_SIZE):
    """Get a subset of NIH training images for BN re-estimation control."""
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

    train_dataset = MalariaDataset(root_dir=DATA_DIR, transform=val_test_transform)
    train_dataset.image_paths = [full_dataset.image_paths[i] for i in train_idx]
    train_dataset.labels = full_dataset.labels[train_idx]

    # Subsample
    rng = np.random.RandomState(RANDOM_SEED)
    if len(train_dataset) > n_samples:
        chosen = rng.choice(len(train_dataset), n_samples, replace=False)
        train_dataset.image_paths = [train_dataset.image_paths[i] for i in chosen]
        train_dataset.labels = train_dataset.labels[chosen]

    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader


def run_inference(model, loader, device):
    """Run inference, return arrays of (probs, labels)."""
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
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def compute_oracle_threshold(probs, labels):
    """Find the threshold that maximises F1 (ORACLE, diagnostic only)."""
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


# ── D1: Calibration vs Ranking ───────────────────────────────────────────

def d1_calibration_ranking(model_type, seed, bbbc041_loader, device):
    """D1: Per-seed calibration analysis."""
    ckpt_path = os.path.join(SCRIPT_DIR,
                              CHECKPOINT_PATTERNS[model_type].format(seed=seed))
    if not os.path.isfile(ckpt_path):
        return None

    build_fn = get_build_fn(model_type)
    model = build_fn().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    probs, labels = run_inference(model, bbbc041_loader, device)

    # Metrics
    auc_roc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    preds_030 = (probs >= THRESHOLD).astype(int)
    frac_positive = preds_030.mean()
    f1_030 = f1_score(labels, preds_030, zero_division=0)

    oracle_thresh, oracle_f1 = compute_oracle_threshold(probs, labels)

    result = {
        'model_type': model_type,
        'seed': seed,
        'bbbc041_auc_roc': float(auc_roc),
        'bbbc041_pr_auc': float(pr_auc),
        'bbbc041_f1_030': float(f1_030),
        'bbbc041_frac_positive_030': float(frac_positive),
        'bbbc041_oracle_threshold': float(oracle_thresh),
        'bbbc041_oracle_f1': float(oracle_f1),
        'collapsed': auc_roc < COLLAPSE_AUC_THRESHOLD,
    }

    # Save probability histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cls, name in [(axes[0], 0, 'Uninfected'), (axes[1], 1, 'Parasitized')]:
        cls_probs = probs[labels == cls]
        ax.hist(cls_probs, bins=HIST_BINS, range=(0, 1), alpha=0.7,
                color='blue' if cls == 0 else 'red', edgecolor='black', linewidth=0.5)
        ax.axvline(x=THRESHOLD, color='green', linestyle='--', label=f'θ={THRESHOLD}')
        ax.axvline(x=oracle_thresh, color='orange', linestyle=':', label=f'ORACLE θ={oracle_thresh:.2f}')
        ax.set_title(f'{name} (n={len(cls_probs)})')
        ax.set_xlabel('P(parasitized)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    fig.suptitle(f'{model_type} seed {seed} — BBBC041 Probability Distribution\n'
                 f'AUC={auc_roc:.3f}, PR-AUC={pr_auc:.3f}, F1@0.30={f1_030:.3f}',
                 fontsize=11)
    fig.tight_layout()
    hist_path = os.path.join(OUTPUT_DIR,
                              f'prob_histogram_{model_type}_seed{seed}.png')
    fig.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result, probs, labels


# ── D2: False-Positive Analysis ──────────────────────────────────────────

def d2_false_positive_analysis(model_type, seed, bbbc041_cells, bbbc041_loader,
                                device):
    """D2: Export top FP crops and analyse correlates."""
    ckpt_path = os.path.join(SCRIPT_DIR,
                              CHECKPOINT_PATTERNS[model_type].format(seed=seed))
    if not os.path.isfile(ckpt_path):
        return None

    build_fn = get_build_fn(model_type)
    model = build_fn().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    probs, labels = run_inference(model, bbbc041_loader, device)

    # Find uninfected cells sorted by probability (highest first = worst FPs)
    uninf_mask = labels == 0
    uninf_indices = np.where(uninf_mask)[0]
    uninf_probs = probs[uninf_mask]
    sorted_order = np.argsort(-uninf_probs)

    top_fp_indices = uninf_indices[sorted_order[:FP_EXPORT_COUNT]]
    top_fp_probs = uninf_probs[sorted_order[:FP_EXPORT_COUNT]]

    # Export image grid
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    fig, axes = plt.subplots(5, 10, figsize=(25, 13))
    fp_details = []

    for i, (cell_idx, prob) in enumerate(zip(top_fp_indices, top_fp_probs)):
        cell_entry = bbbc041_cells[cell_idx]
        img_path, bbox, label = cell_entry

        # Load and crop
        try:
            img = Image.open(img_path).convert('RGB')
            pad_x = int(bbox['w'] * 0.10)
            pad_y = int(bbox['h'] * 0.10)
            x1 = max(0, bbox['x'] - pad_x)
            y1 = max(0, bbox['y'] - pad_y)
            x2 = min(img.width, bbox['x'] + bbox['w'] + pad_x)
            y2 = min(img.height, bbox['y'] + bbox['h'] + pad_y)
            crop = img.crop((x1, y1, x2, y2))
            crop_w, crop_h = crop.size
        except Exception:
            crop = Image.new('RGB', (64, 64), (0, 0, 0))
            crop_w, crop_h = 64, 64

        row, col = i // 10, i % 10
        axes[row, col].imshow(crop)
        axes[row, col].set_title(f'p={prob:.2f}\n{crop_w}×{crop_h}', fontsize=7)
        axes[row, col].axis('off')

        fp_details.append({
            'rank': i,
            'cell_idx': int(cell_idx),
            'prob': float(prob),
            'bbox_w': bbox['w'],
            'bbox_h': bbox['h'],
            'source_image': os.path.basename(img_path),
        })

    fig.suptitle(f'{model_type} seed {seed} — Top {FP_EXPORT_COUNT} False Positives (Uninfected → P(para))',
                 fontsize=12)
    fig.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, f'fp_grid_{model_type}_seed{seed}.png')
    fig.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Analyse correlations
    bbox_ws = [d['bbox_w'] for d in fp_details]
    bbox_hs = [d['bbox_h'] for d in fp_details]
    source_imgs = [d['source_image'] for d in fp_details]

    # Correlation with crop size
    if len(set(top_fp_probs)) > 1:
        corr_w, pval_w = stats.spearmanr(top_fp_probs, bbox_ws)
        corr_h, pval_h = stats.spearmanr(top_fp_probs, bbox_hs)
    else:
        corr_w, pval_w = 0.0, 1.0
        corr_h, pval_h = 0.0, 1.0

    # Source image concentration
    from collections import Counter
    source_counts = Counter(source_imgs)
    n_unique_sources = len(source_counts)
    most_common_source = source_counts.most_common(1)[0] if source_counts else ('', 0)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'model_type': model_type,
        'seed': seed,
        'fp_count': FP_EXPORT_COUNT,
        'corr_w_spearman': float(corr_w),
        'corr_w_pvalue': float(pval_w),
        'corr_h_spearman': float(corr_h),
        'corr_h_pvalue': float(pval_h),
        'n_unique_source_images': n_unique_sources,
        'most_common_source': most_common_source[0],
        'most_common_source_count': most_common_source[1],
        'grid_path': grid_path,
    }


# ── D3: Attention Gate Statistics ─────────────────────────────────────────

def collect_gate_stats_cbam(model, loader, device, n_samples=GATE_SAMPLE_SIZE):
    """
    Collect CBAM gate statistics using forward hooks.
    Returns dict with per-stage stats.
    """
    stage_stats = {}
    hooks = []
    gate_data = {}  # stage_idx -> {'ca': [], 'sa': [], 'input_mag': [], 'output_mag': []}

    for idx in [5, 6, 7]:
        stage = model.features[idx]
        # stage is nn.Sequential(mbconv_block, CBAM)
        cbam = stage[1]  # The CBAM module

        gate_data[idx] = {
            'ca_gates': [],
            'sa_gates': [],
            'input_mag': [],
            'output_mag': [],
        }

        # Hook on the CBAM module itself
        def make_cbam_hook(stage_idx):
            def hook_fn(module, input, output):
                with torch.no_grad():
                    x_in = input[0].detach()
                    x_out = output.detach()

                    # Channel attention gate
                    ca_out = module.sigmoid(module.ca(x_in))  # (B, C, 1, 1)
                    # Spatial attention: needs post-channel-attention input
                    x_after_ca = x_in * ca_out
                    sa_out = module.sigmoid(module.sa(x_after_ca))  # (B, 1, H, W)

                    gate_data[stage_idx]['ca_gates'].append(ca_out.cpu())
                    gate_data[stage_idx]['sa_gates'].append(sa_out.cpu())
                    gate_data[stage_idx]['input_mag'].append(
                        x_in.abs().mean(dim=(1, 2, 3)).cpu())
                    gate_data[stage_idx]['output_mag'].append(
                        x_out.abs().mean(dim=(1, 2, 3)).cpu())
            return hook_fn

        hooks.append(cbam.register_forward_hook(make_cbam_hook(idx)))

    # Run inference on sample
    model.eval()
    n_collected = 0
    with torch.no_grad():
        for images, labels in loader:
            if n_collected >= n_samples:
                break
            batch_size = min(images.shape[0], n_samples - n_collected)
            images = images[:batch_size].to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    model(images)
            else:
                model(images)
            n_collected += batch_size

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate statistics
    for idx in [5, 6, 7]:
        ca = torch.cat(gate_data[idx]['ca_gates'], dim=0)  # (N, C, 1, 1)
        sa = torch.cat(gate_data[idx]['sa_gates'], dim=0)  # (N, 1, H, W)
        in_mag = torch.cat(gate_data[idx]['input_mag'])
        out_mag = torch.cat(gate_data[idx]['output_mag'])

        ca_flat = ca.squeeze().numpy()  # (N, C)
        sa_flat = sa.squeeze().numpy()  # (N, H, W) or (N, H*W)
        if sa_flat.ndim == 3:
            sa_flat = sa_flat.reshape(sa_flat.shape[0], -1)

        stage_stats[idx] = {
            'ca_mean': float(ca_flat.mean()),
            'ca_sd': float(ca_flat.std()),
            'ca_frac_saturated_high': float((ca_flat > GATE_SATURATION_HIGH).mean()),
            'ca_frac_saturated_low': float((ca_flat < GATE_SATURATION_LOW).mean()),
            'sa_mean': float(sa_flat.mean()),
            'sa_sd': float(sa_flat.std()),
            'sa_frac_saturated_high': float((sa_flat > GATE_SATURATION_HIGH).mean()),
            'sa_frac_saturated_low': float((sa_flat < GATE_SATURATION_LOW).mean()),
            'input_mag_mean': float(in_mag.numpy().mean()),
            'input_mag_sd': float(in_mag.numpy().std()),
            'output_mag_mean': float(out_mag.numpy().mean()),
            'output_mag_sd': float(out_mag.numpy().std()),
            'n_samples': int(ca.shape[0]),
        }

    return stage_stats


def collect_gate_stats_eca(model, loader, device, n_samples=GATE_SAMPLE_SIZE):
    """
    Collect ECA gate statistics using forward hooks.
    ECA has a single channel gate (sigmoid output), no spatial attention.
    """
    stage_stats = {}
    hooks = []
    gate_data = {}

    for idx in [5, 6, 7]:
        stage = model.features[idx]
        eca = stage[1]  # The ECALayer module

        gate_data[idx] = {
            'gate': [],
            'input_mag': [],
            'output_mag': [],
        }

        def make_eca_hook(stage_idx):
            def hook_fn(module, input, output):
                with torch.no_grad():
                    x_in = input[0].detach()
                    x_out = output.detach()

                    # Recompute the gate
                    y = module.avg_pool(x_in)
                    y = y.squeeze(-1).transpose(-1, -2)
                    y = module.conv(y)
                    y = y.transpose(-1, -2).unsqueeze(-1)
                    gate = module.sigmoid(y)  # (B, C, 1, 1)

                    gate_data[stage_idx]['gate'].append(gate.cpu())
                    gate_data[stage_idx]['input_mag'].append(
                        x_in.abs().mean(dim=(1, 2, 3)).cpu())
                    gate_data[stage_idx]['output_mag'].append(
                        x_out.abs().mean(dim=(1, 2, 3)).cpu())
            return hook_fn

        hooks.append(eca.register_forward_hook(make_eca_hook(idx)))

    model.eval()
    n_collected = 0
    with torch.no_grad():
        for images, labels in loader:
            if n_collected >= n_samples:
                break
            batch_size = min(images.shape[0], n_samples - n_collected)
            images = images[:batch_size].to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    model(images)
            else:
                model(images)
            n_collected += batch_size

    for h in hooks:
        h.remove()

    for idx in [5, 6, 7]:
        gates = torch.cat(gate_data[idx]['gate'], dim=0).squeeze().numpy()
        in_mag = torch.cat(gate_data[idx]['input_mag']).numpy()
        out_mag = torch.cat(gate_data[idx]['output_mag']).numpy()

        stage_stats[idx] = {
            'gate_mean': float(gates.mean()),
            'gate_sd': float(gates.std()),
            'gate_frac_saturated_high': float((gates > GATE_SATURATION_HIGH).mean()),
            'gate_frac_saturated_low': float((gates < GATE_SATURATION_LOW).mean()),
            'input_mag_mean': float(in_mag.mean()),
            'input_mag_sd': float(in_mag.std()),
            'output_mag_mean': float(out_mag.mean()),
            'output_mag_sd': float(out_mag.std()),
            'n_samples': int(gates.shape[0]),
        }

    return stage_stats


def d3_gate_statistics(device, bbbc041_loader, nih_loader):
    """D3: Collect and compare gate statistics across datasets and seeds."""
    all_gate_rows = []

    for model_type in ['hybrid_cbam', 'hybrid_eca']:
        collect_fn = collect_gate_stats_cbam if model_type == 'hybrid_cbam' else collect_gate_stats_eca

        for seed in SEEDS:
            ckpt_path = os.path.join(SCRIPT_DIR,
                                      CHECKPOINT_PATTERNS[model_type].format(seed=seed))
            if not os.path.isfile(ckpt_path):
                print(f"    [SKIP] {model_type} seed {seed}: checkpoint not found")
                continue

            build_fn = get_build_fn(model_type)
            model = build_fn().to(device)
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)

            for dataset_name, loader in [('NIH', nih_loader), ('BBBC041', bbbc041_loader)]:
                print(f"    Gate stats: {model_type} seed {seed} on {dataset_name}")
                gate_stats = collect_fn(model, loader, device)

                for stage_idx, ss in gate_stats.items():
                    row = {
                        'model_type': model_type,
                        'seed': seed,
                        'dataset': dataset_name,
                        'stage': stage_idx,
                    }
                    row.update(ss)
                    all_gate_rows.append(row)

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return all_gate_rows


# ── D4: BatchNorm Re-estimation ──────────────────────────────────────────

def reset_bn_stats(model):
    """Reset all BatchNorm running statistics to cumulative average mode."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative average
            m.train()  # so running stats update


def set_bn_only_train(model):
    """
    Put only BatchNorm layers in train mode (for running stat update).
    Everything else stays in eval mode (dropout off, no gradients).
    """
    model.eval()  # First set everything to eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()


def run_bn_reestimation(model, reest_loader, eval_loader, device):
    """
    Reset BN stats, re-estimate on reest_loader, evaluate on eval_loader.
    Returns metrics dict.
    """
    set_bn_only_train(model)

    # Forward pass on reest data (no gradients, no labels)
    with torch.no_grad():
        for images, _ in reest_loader:
            images = images.to(device)
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    model(images)
            else:
                model(images)

    # Now evaluate
    model.eval()  # Set BN back to eval (use the re-estimated stats)
    probs, labels = run_inference(model, eval_loader, device)

    auc = roc_auc_score(labels, probs)
    preds = (probs >= THRESHOLD).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)

    return {'auc_roc': float(auc), 'f1': float(f1)}


def d4_bn_reestimation(device, bbbc041_loader, bbbc041_train_cells,
                        nih_train_loader):
    """D4: BatchNorm re-estimation diagnostic."""
    import hashlib

    sys.path.insert(0, SCRIPT_DIR)
    from bbbc041_eval import BBBC041Dataset

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create BBBC041 training loader (unlabelled — we just ignore labels)
    rng = np.random.RandomState(RANDOM_SEED)
    if len(bbbc041_train_cells) > BN_REESTIMATION_SIZE:
        chosen = rng.choice(len(bbbc041_train_cells), BN_REESTIMATION_SIZE, replace=False)
        bn_cells = [bbbc041_train_cells[i] for i in chosen]
    else:
        bn_cells = bbbc041_train_cells

    bn_dataset = BBBC041Dataset(bn_cells, transform=val_test_transform)
    bn_loader = DataLoader(bn_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    results = []

    for model_type in ['hybrid_cbam', 'hybrid_eca', 'baseline']:
        for seed in SEEDS:
            ckpt_path = os.path.join(SCRIPT_DIR,
                                      CHECKPOINT_PATTERNS[model_type].format(seed=seed))
            if not os.path.isfile(ckpt_path):
                continue

            build_fn = get_build_fn(model_type)
            sha256 = ''
            if model_type == 'hybrid_cbam' and seed == 42:
                h = hashlib.sha256()
                with open(ckpt_path, 'rb') as f:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        h.update(chunk)
                sha256 = h.hexdigest()

            # ── Before: original BN stats ──
            print(f"    BN re-est: {model_type} seed {seed} — evaluating original...")
            model = build_fn().to(device)
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            probs_before, labels_before = run_inference(model, bbbc041_loader, device)
            auc_before = roc_auc_score(labels_before, probs_before)
            preds_before = (probs_before >= THRESHOLD).astype(int)
            f1_before = f1_score(labels_before, preds_before, zero_division=0)

            del model

            # ── After: re-estimated on BBBC041 training ──
            print(f"    BN re-est: {model_type} seed {seed} — re-estimating on BBBC041 train...")
            model = build_fn().to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True))
            reset_bn_stats(model)
            after_bbbc = run_bn_reestimation(model, bn_loader, bbbc041_loader, device)
            del model

            # ── Control: re-estimated on NIH training ──
            print(f"    BN re-est: {model_type} seed {seed} — control (NIH train)...")
            model = build_fn().to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True))

            # First evaluate on BBBC041 with original stats as baseline
            # Then re-estimate on NIH train and evaluate on BBBC041
            reset_bn_stats(model)
            after_nih = run_bn_reestimation(model, nih_train_loader, bbbc041_loader, device)
            del model

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            results.append({
                'model_type': model_type,
                'seed': seed,
                'sha256': sha256,
                'bbbc041_auc_before': auc_before,
                'bbbc041_f1_before': f1_before,
                'bbbc041_auc_after_bbbc041_bn': after_bbbc['auc_roc'],
                'bbbc041_f1_after_bbbc041_bn': after_bbbc['f1'],
                'bbbc041_auc_after_nih_bn': after_nih['auc_roc'],
                'bbbc041_f1_after_nih_bn': after_nih['f1'],
            })

    return results


# ── D5 (Optional): CBAM Bypass ───────────────────────────────────────────

def d5_cbam_bypass(device, bbbc041_loader, nih_loader):
    """
    D5: Inference-time bypass of attention gates.
    CAVEAT: downstream layers were trained with attention active.
    """
    results = []

    for model_type in ['hybrid_cbam', 'hybrid_eca']:
        for seed in SEEDS:
            ckpt_path = os.path.join(SCRIPT_DIR,
                                      CHECKPOINT_PATTERNS[model_type].format(seed=seed))
            if not os.path.isfile(ckpt_path):
                continue

            build_fn = get_build_fn(model_type)

            for bypass_mode in ['channel_only', 'spatial_only', 'both']:
                for dataset_name, loader in [('NIH', nih_loader), ('BBBC041', bbbc041_loader)]:
                    model = build_fn().to(device)
                    model.load_state_dict(
                        torch.load(ckpt_path, map_location=device, weights_only=True))
                    model.eval()

                    # Patch the attention modules
                    if model_type == 'hybrid_cbam':
                        for idx in [5, 6, 7]:
                            cbam = model.features[idx][1]
                            if bypass_mode in ['channel_only', 'both']:
                                # Force channel gate to 1
                                cbam.ca = _ConstantGate(1.0)
                            if bypass_mode in ['spatial_only', 'both']:
                                # Force spatial gate to 1
                                cbam.sa = _ConstantSpatialGate(1.0)
                    elif model_type == 'hybrid_eca':
                        for idx in [5, 6, 7]:
                            eca = model.features[idx][1]
                            if bypass_mode in ['channel_only', 'both']:
                                # For ECA, bypass = identity (gate = 1)
                                eca.forward = lambda self, x: x
                                # Patch more cleanly
                                model.features[idx][1] = _IdentityModule()

                    probs, labels = run_inference(model, loader, device)
                    auc = roc_auc_score(labels, probs)
                    preds = (probs >= THRESHOLD).astype(int)
                    f1 = f1_score(labels, preds, zero_division=0)

                    results.append({
                        'model_type': model_type,
                        'seed': seed,
                        'bypass_mode': bypass_mode,
                        'dataset': dataset_name,
                        'auc_roc': float(auc),
                        'f1': float(f1),
                        'caveat': 'downstream layers trained with attention active',
                    })

                    del model
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    return results


class _ConstantGate(nn.Module):
    """Replace channel attention with constant gate value."""
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return torch.ones_like(x) * self.val


class _ConstantSpatialGate(nn.Module):
    """Replace spatial attention conv with constant output."""
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        # Spatial attention input is (B, 2, H, W), output should be (B, 1, H, W)
        # But we're replacing sa, which gets called with the post-CA tensor
        # CBAM.forward does: sa_out = self.sigmoid(self.sa(x))
        # self.sa.forward(x) normally does concat + conv → (B, 1, H, W)
        # We need to return pre-sigmoid value that gives gate=1: large positive
        return torch.ones(x.shape[0], 1, x.shape[2], x.shape[3],
                          device=x.device, dtype=x.dtype) * 10.0  # sigmoid(10) ≈ 1


class _IdentityModule(nn.Module):
    """Identity module for ECA bypass."""
    def forward(self, x):
        return x


# ── D6: Seed 42 Verification ─────────────────────────────────────────────

def d6_verify_seed42(device, nih_loader, bbbc041_loader):
    """D6: Verify seed 42 hybrid checkpoint independently."""
    import hashlib

    ckpt_path = os.path.join(SCRIPT_DIR, 'best_model_seed42.pt')
    if not os.path.isfile(ckpt_path):
        print("  [SKIP] Hybrid CBAM seed 42 checkpoint not found")
        return None

    # SHA256
    h = hashlib.sha256()
    with open(ckpt_path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    sha = h.hexdigest()

    build_fn = get_build_fn('hybrid_cbam')
    model = build_fn().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    # NIH test
    nih_probs, nih_labels = run_inference(model, nih_loader, device)
    nih_auc = roc_auc_score(nih_labels, nih_probs)
    nih_preds = (nih_probs >= THRESHOLD).astype(int)
    nih_f1 = f1_score(nih_labels, nih_preds, zero_division=0)

    # BBBC041 test
    bbbc_probs, bbbc_labels = run_inference(model, bbbc041_loader, device)
    bbbc_auc = roc_auc_score(bbbc_labels, bbbc_probs)
    bbbc_preds = (bbbc_probs >= THRESHOLD).astype(int)
    bbbc_f1 = f1_score(bbbc_labels, bbbc_preds, zero_division=0)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'sha256': sha,
        'nih_auc': float(nih_auc),
        'nih_f1': float(nih_f1),
        'bbbc041_auc': float(bbbc_auc),
        'bbbc041_f1': float(bbbc_f1),
    }


# ── Plotting helpers ─────────────────────────────────────────────────────

def plot_gate_comparison(gate_rows, output_dir):
    """Plot gate statistics comparison across seeds and datasets."""
    import pandas as pd
    df = pd.DataFrame(gate_rows)

    for model_type in df['model_type'].unique():
        mdf = df[df['model_type'] == model_type]

        if model_type == 'hybrid_cbam':
            metrics = ['ca_mean', 'sa_mean', 'ca_frac_saturated_high', 'sa_frac_saturated_high']
            titles = ['Channel Gate Mean', 'Spatial Gate Mean',
                      'Channel Gate Frac Saturated (>0.95)', 'Spatial Gate Frac Saturated (>0.95)']
        else:
            metrics = ['gate_mean', 'gate_frac_saturated_high', 'input_mag_mean', 'output_mag_mean']
            titles = ['Gate Mean', 'Gate Frac Saturated (>0.95)',
                      'Input Magnitude Mean', 'Output Magnitude Mean']

        available_metrics = [m for m in metrics if m in mdf.columns]
        available_titles = [titles[metrics.index(m)] for m in available_metrics]

        if not available_metrics:
            continue

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 3, figsize=(15, 4 * n_metrics))
        if n_metrics == 1:
            axes = axes.reshape(1, -1)

        for mi, (metric, title) in enumerate(zip(available_metrics, available_titles)):
            for si, stage in enumerate([5, 6, 7]):
                ax = axes[mi, si]
                stage_data = mdf[mdf['stage'] == stage]

                for dataset_name, color in [('NIH', 'blue'), ('BBBC041', 'red')]:
                    ds_data = stage_data[stage_data['dataset'] == dataset_name]
                    seeds_vals = ds_data.sort_values('seed')
                    ax.bar(
                        [f"s{s}" for s in seeds_vals['seed']],
                        seeds_vals[metric],
                        alpha=0.6, label=dataset_name, color=color,
                        width=0.35,
                        # Offset bars
                    )

                # Highlight seed 42
                ax.set_title(f'{title} — Stage {stage}', fontsize=9)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

        fig.suptitle(f'{model_type} — Gate Statistics: NIH vs BBBC041', fontsize=12)
        fig.tight_layout()
        path = os.path.join(output_dir, f'gate_comparison_{model_type}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_bn_reestimation(bn_results, output_dir):
    """Plot BN re-estimation before/after comparison."""
    import pandas as pd
    df = pd.DataFrame(bn_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in [(axes[0], 'auc', 'AUC-ROC'), (axes[1], 'f1', 'F1-Score')]:
        x = np.arange(len(df))
        labels = [f"{r['model_type']}\ns{r['seed']}" for _, r in df.iterrows()]

        before = df[f'bbbc041_{metric}_before'].values
        after_bbbc = df[f'bbbc041_{metric}_after_bbbc041_bn'].values
        after_nih = df[f'bbbc041_{metric}_after_nih_bn'].values

        w = 0.25
        ax.bar(x - w, before, w, label='Original BN', alpha=0.8, color='steelblue')
        ax.bar(x, after_bbbc, w, label='Re-est. BBBC041 train', alpha=0.8, color='coral')
        ax.bar(x + w, after_nih, w, label='Re-est. NIH train (ctrl)', alpha=0.8, color='gray')

        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.set_title(f'BBBC041 {title}: BN Re-estimation')

    fig.tight_layout()
    path = os.path.join(output_dir, 'bn_reestimation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_collapse_paper_table(d1_results, bn_results, d6_result):
    """Generate paper table for collapse diagnostics."""
    lines = ["\n# BBBC041 Seed-Collapse Diagnostics\n"]
    lines.append(f"Collapse rule: AUC-ROC < {COLLAPSE_AUC_THRESHOLD} (pre-specified)\n")
    lines.append(f"Seeds: {SEEDS}\n")

    # D1 table
    if d1_results:
        lines.append("## D1: Calibration and Ranking\n")
        lines.append("| Model | Seed | AUC-ROC | PR-AUC | F1@0.30 | Frac Pos@0.30 | ORACLE θ | ORACLE F1 | Collapsed? |")
        lines.append("|-------|------|---------|--------|---------|--------------|----------|-----------|-----------|")
        for r in d1_results:
            collapsed = "YES" if r['collapsed'] else "no"
            lines.append(
                f"| {r['model_type']} | {r['seed']} | {r['bbbc041_auc_roc']:.3f} | "
                f"{r['bbbc041_pr_auc']:.3f} | {r['bbbc041_f1_030']:.3f} | "
                f"{r['bbbc041_frac_positive_030']:.3f} | "
                f"{r['bbbc041_oracle_threshold']:.2f} | {r['bbbc041_oracle_f1']:.3f} | "
                f"{collapsed} |"
            )
        # Count collapses
        collapse_counts = {}
        for r in d1_results:
            key = r['model_type']
            if key not in collapse_counts:
                collapse_counts[key] = {'total': 0, 'collapsed': 0}
            collapse_counts[key]['total'] += 1
            if r['collapsed']:
                collapse_counts[key]['collapsed'] += 1
        lines.append("")
        lines.append("Collapse summary:")
        for mt, counts in collapse_counts.items():
            lines.append(f"  - {mt}: {counts['collapsed']}/{counts['total']} seeds collapsed")
        lines.append("")

    # D4 BN re-estimation table
    if bn_results:
        lines.append("## D4: BatchNorm Re-estimation\n")
        lines.append("| Model | Seed | AUC (orig) | AUC (BBBC041 BN) | AUC (NIH BN ctrl) | F1 (orig) | F1 (BBBC041 BN) | F1 (NIH BN ctrl) |")
        lines.append("|-------|------|-----------|-----------------|------------------|----------|----------------|-----------------|")
        for r in bn_results:
            lines.append(
                f"| {r['model_type']} | {r['seed']} | {r['bbbc041_auc_before']:.3f} | "
                f"{r['bbbc041_auc_after_bbbc041_bn']:.3f} | {r['bbbc041_auc_after_nih_bn']:.3f} | "
                f"{r['bbbc041_f1_before']:.3f} | {r['bbbc041_f1_after_bbbc041_bn']:.3f} | "
                f"{r['bbbc041_f1_after_nih_bn']:.3f} |"
            )
        lines.append("")

    # D6 verification
    if d6_result:
        lines.append("## D6: Seed-42 Hybrid (CBAM) Checkpoint Verification\n")
        lines.append(f"- SHA256: `{d6_result['sha256']}`")
        lines.append(f"- NIH test AUC: {d6_result['nih_auc']:.4f}, F1: {d6_result['nih_f1']:.4f}")
        lines.append(f"- BBBC041 test AUC: {d6_result['bbbc041_auc']:.4f}, F1: {d6_result['bbbc041_f1']:.4f}")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nRNG seed: {RANDOM_SEED}")
    print(f"Collapse rule: AUC-ROC < {COLLAPSE_AUC_THRESHOLD}")
    print(f"Seeds: {SEEDS}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Loading datasets...")
    print("=" * 60)

    bbbc041_cells, bbbc041_loader = get_bbbc041_test_data()
    nih_loader, nih_test_dataset = get_nih_test_loader()
    nih_train_loader = get_nih_train_loader()
    bbbc041_train_cells = get_bbbc041_training_data()

    print(f"  BBBC041 test cells: {len(bbbc041_cells)}")
    print(f"  BBBC041 train cells: {len(bbbc041_train_cells)}")

    # ── D1: Calibration vs Ranking ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  D1: Calibration and Ranking Analysis")
    print("=" * 60)

    d1_results = []
    for model_type in ['hybrid_cbam', 'hybrid_eca', 'baseline']:
        for seed in SEEDS:
            print(f"\n  {model_type} seed {seed}...")
            result = d1_calibration_ranking(model_type, seed, bbbc041_loader, device)
            if result:
                d1_results.append(result[0])
                print(f"    AUC={result[0]['bbbc041_auc_roc']:.3f}, "
                      f"F1@0.30={result[0]['bbbc041_f1_030']:.3f}, "
                      f"ORACLE θ={result[0]['bbbc041_oracle_threshold']:.2f}, "
                      f"Collapsed={'YES' if result[0]['collapsed'] else 'no'}")

    # ── D2: False-Positive Analysis ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  D2: False-Positive Analysis")
    print("=" * 60)

    d2_results = []
    for model_type in ['hybrid_cbam']:  # Focus on CBAM since that's the collapse model
        for seed in SEEDS:
            print(f"\n  {model_type} seed {seed}...")
            result = d2_false_positive_analysis(
                model_type, seed, bbbc041_cells, bbbc041_loader, device)
            if result:
                d2_results.append(result)
                print(f"    FP size corr (w): r={result['corr_w_spearman']:.3f}, "
                      f"p={result['corr_w_pvalue']:.3f}")
                print(f"    Unique source images: {result['n_unique_source_images']}")

    # ── D3: Gate Statistics ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  D3: Attention Gate Statistics")
    print("=" * 60)

    gate_rows = d3_gate_statistics(device, bbbc041_loader, nih_loader)

    if gate_rows:
        import pandas as pd
        gate_csv = os.path.join(OUTPUT_DIR, 'gate_statistics.csv')
        pd.DataFrame(gate_rows).to_csv(gate_csv, index=False)
        print(f"\n  Saved: {gate_csv}")

        plot_gate_comparison(gate_rows, OUTPUT_DIR)
        print("  Saved gate comparison plots")

    # ── D4: BatchNorm Re-estimation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  D4: BatchNorm Re-estimation Diagnostic")
    print("=" * 60)

    bn_results = d4_bn_reestimation(
        device, bbbc041_loader, bbbc041_train_cells, nih_train_loader)

    if bn_results:
        import pandas as pd
        bn_csv = os.path.join(OUTPUT_DIR, 'bn_reestimation.csv')
        pd.DataFrame(bn_results).to_csv(bn_csv, index=False)
        print(f"\n  Saved: {bn_csv}")

        plot_bn_reestimation(bn_results, OUTPUT_DIR)
        print("  Saved BN re-estimation plot")

    # ── D5: CBAM Bypass (optional) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  D5 (Optional): Attention Bypass Diagnostic")
    print("=" * 60)

    try:
        bypass_results = d5_cbam_bypass(device, bbbc041_loader, nih_loader)
        if bypass_results:
            import pandas as pd
            bypass_csv = os.path.join(OUTPUT_DIR, 'attention_bypass.csv')
            pd.DataFrame(bypass_results).to_csv(bypass_csv, index=False)
            print(f"\n  Saved: {bypass_csv}")
    except Exception as e:
        print(f"  [WARNING] D5 bypass failed: {e}")
        bypass_results = []

    # ── D6: Seed 42 Verification ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  D6: Seed-42 Hybrid Checkpoint Verification")
    print("=" * 60)

    d6_result = d6_verify_seed42(device, nih_loader, bbbc041_loader)
    if d6_result:
        print(f"  SHA256: {d6_result['sha256'][:32]}...")
        print(f"  NIH:    AUC={d6_result['nih_auc']:.4f}, F1={d6_result['nih_f1']:.4f}")
        print(f"  BBBC041: AUC={d6_result['bbbc041_auc']:.4f}, F1={d6_result['bbbc041_f1']:.4f}")

    # ── Save combined diagnostics CSV ─────────────────────────────────
    import pandas as pd
    if d1_results:
        diag_csv = os.path.join(OUTPUT_DIR, 'bbbc041_diagnostics_per_seed.csv')
        pd.DataFrame(d1_results).to_csv(diag_csv, index=False)
        print(f"\nSaved: {diag_csv}")

    # ── Generate paper tables ─────────────────────────────────────────
    collapse_text = generate_collapse_paper_table(d1_results, bn_results, d6_result)

    # Merge with existing Grad-CAM tables if they exist
    paper_path = os.path.join(OUTPUT_DIR, 'paper_tables.md')
    existing_text = ""
    gradcam_path = os.path.join(OUTPUT_DIR, 'gradcam_paper_tables.md')
    if os.path.isfile(gradcam_path):
        with open(gradcam_path, 'r', encoding='utf-8') as f:
            existing_text = f.read()

    with open(paper_path, 'w', encoding='utf-8') as f:
        if existing_text:
            f.write(existing_text)
            f.write("\n---\n")
        f.write(collapse_text)

    print(f"Saved: {paper_path}")

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE — Summary")
    print("=" * 60)
    if d1_results:
        n_collapsed = sum(1 for r in d1_results if r['collapsed'])
        n_total = len(d1_results)
        print(f"  Collapse count: {n_collapsed}/{n_total} seed/model combinations")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
