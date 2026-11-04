"""
Phase 1: Quantitative Grad-CAM Evaluation
==========================================
Computes faithfulness and concentration metrics for Grad-CAM heatmaps over
ALL 1,378 parasitized cells in the NIH test set, for every model type, seed,
and target layer. Includes random and centre-prior controls, plus optional
proxy parasite pseudo-mask analysis.

Analysis choices (cite these in the methods section):
  - Grad-CAM: forward-hooked activation + torch.autograd.grad (batched)
  - Heatmaps: 7×7 → 224×224 bilinear upsampling, min-max normalised [0,1]
  - Pixel ranking: stable argsort with deterministic index tie-breaking
  - Deletion AUC: replace most-salient pixels first with 0 (normalised space),
    steps of 5% (21 steps: 0%, 5%, ..., 100%), trapezoid integration
  - Insertion AUC: start from Gaussian-blurred image (σ=10 in pixel space),
    reveal most-salient pixels, same step schedule, trapezoid integration
  - Normalised Shannon entropy: H / log(N_pixels), heatmap as probability dist.
  - Top-10% mass: fraction of total heat in top 10% most-salient pixels
  - On-cell mass: cell mask by intensity threshold ≥30 on de-normalised uint8
    grayscale image (cells are stained, background is near-black)
  - Random control RNG seed: 12345
  - Centre-prior: isotropic Gaussian blob at (112,112) with σ=40
  - Insertion blur σ: 10 pixels (applied to normalised image channels)
  - Colour-threshold parasite pseudo-mask (optional proxy):
    Purple/dark-stained region in HSV space: H ∈ [120,170], S ∈ [30,255], V ∈ [20,180]
    (Giemsa stain makes parasites appear as dark purple inclusions)
  - Batch size for Grad-CAM: 8 (to fit in 6 GB VRAM with fp32)
  - fp32 throughout (no autocast)

Outputs (in analysis_outputs/):
  - gradcam_metrics_per_cell.csv     (one row per cell × model × layer)
  - gradcam_metrics_per_seed.csv     (averaged within each seed)
  - gradcam_metrics_summary.csv      (mean ± SD across seeds)
  - gradcam_proxy_masks/             (optional: 100 TP crops per model)

Usage:
    python gradcam_metrics.py                    # full run
    python gradcam_metrics.py --max-cells 50     # quick dev run

RNG seeds used: 12345 (random control + sampling)
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy import stats, ndimage
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 2026, 7, 17]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis_outputs')

THRESHOLD = 0.30
GRADCAM_BATCH_SIZE = 8   # fp32, ~6 GB VRAM headroom
DATA_BATCH_SIZE = 32
NUM_WORKERS = 2

# Grad-CAM analysis parameters
DELETION_STEPS = np.arange(0, 1.05, 0.05)  # 0%, 5%, ..., 100%
INSERTION_BLUR_SIGMA = 10.0
CELL_MASK_THRESHOLD = 30  # intensity on de-normalised uint8 grayscale
TOP_K_FRACTION = 0.10     # top-10% mass
RANDOM_SEED = 12345
CENTRE_SIGMA = 40.0

# Proxy mask (optional) HSV thresholds for Giemsa-purple parasite stain
PROXY_H_RANGE = (120, 170)  # Hue
PROXY_S_RANGE = (30, 255)   # Saturation
PROXY_V_RANGE = (20, 180)   # Value

# Checkpoint patterns
CHECKPOINT_PATTERNS = {
    'hybrid_cbam': 'best_model_seed{seed}.pt',
    'hybrid_eca':  'best_model_ecanet_seed{seed}.pt',
    'baseline':    'checkpoint_baseline_seed_{seed}.pth',
}

# Layer configurations: (model_type, layer_label, layer_accessor)
# layer_accessor is a function: model -> target_layer module
LAYER_CONFIGS = [
    ('baseline',    'features[7]',          lambda m: m.features[7]),
    ('hybrid_cbam', 'features[7][0] (pre-CBAM)',  lambda m: m.features[7][0]),
    ('hybrid_cbam', 'features[7] (post-CBAM)',    lambda m: m.features[7]),
    ('hybrid_eca',  'features[7][0] (pre-ECA)',   lambda m: m.features[7][0]),
    ('hybrid_eca',  'features[7] (post-ECA)',     lambda m: m.features[7]),
]


# ── ImageNet normalisation constants ──────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def get_build_fn(model_type):
    """Import model builder dynamically."""
    sys.path.insert(0, SCRIPT_DIR)
    module_map = {
        'hybrid_cbam': 'model',
        'hybrid_eca':  'model_ecanet',
        'baseline':    'model_baseline',
    }
    mod = __import__(module_map[model_type])
    return mod.build_model


def get_test_data():
    """Return test dataset (not loader) with the exact split."""
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

    return test_dataset, test_idx, full_dataset


# ── Grad-CAM Implementation (batched, using autograd.grad) ────────────────

class BatchGradCAM:
    """
    Batched Grad-CAM using forward hooks + torch.autograd.grad.
    No backward hooks needed — more reliable and explicit.
    """

    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self._hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        # Keep the activation in the graph (do NOT detach)
        self.activations = output

    def generate_batch(self, images):
        """
        Generate Grad-CAM heatmaps for a batch.

        Args:
            images: (B, 3, 224, 224) tensor on device, requires_grad not needed

        Returns:
            heatmaps: (B, 224, 224) numpy array, min-max normalised to [0,1]
        """
        self.model.eval()
        # We need gradients w.r.t. the activations, not the input
        # The forward hook captures activations WITH the computation graph intact

        # Forward pass
        self.activations = None
        logits = self.model(images)  # (B, 1)

        if self.activations is None:
            raise RuntimeError("Forward hook did not fire on the target layer.")

        acts = self.activations  # (B, C, H, W) — still in the graph

        # Compute gradients of each logit w.r.t. the activations
        # We use the sum of logits as the scalar target (equivalent to
        # computing grad for each sample separately because they are independent)
        score = logits.squeeze(-1).sum()  # scalar

        grads = torch.autograd.grad(
            outputs=score,
            inputs=acts,
            create_graph=False,
            retain_graph=False,
        )[0]  # (B, C, H, W)

        # GAP of gradients → channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination + ReLU
        cam = (weights * acts.detach()).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)

        # Upsample to input size
        cam = cam.unsqueeze(1)  # (B, 1, H, W)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)  # (B, 224, 224)

        # Min-max normalise per sample
        heatmaps = cam.cpu().numpy()
        for i in range(heatmaps.shape[0]):
            h = heatmaps[i]
            hmin, hmax = h.min(), h.max()
            if hmax > 0:
                h = (h - hmin) / (hmax - hmin)
            else:
                h = np.zeros_like(h)
            heatmaps[i] = h

        return heatmaps

    def remove_hook(self):
        self._hook.remove()


def verify_against_existing_gradcam(model, target_layer, test_dataset, device, n_check=10):
    """
    Verify that our batched implementation reproduces the existing gradcam.py
    heatmaps on at least n_check cells.
    Returns max absolute difference.
    """
    sys.path.insert(0, SCRIPT_DIR)
    from gradcam import GradCAM as OldGradCAM

    # Find parasitized cells
    para_indices = [i for i in range(len(test_dataset))
                    if test_dataset.labels[i].item() == 1][:n_check]

    old_gc = OldGradCAM(model, target_layer=target_layer)
    new_gc = BatchGradCAM(model, target_layer, device)

    max_diff = 0.0
    for idx in para_indices:
        img, label = test_dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        # Old method (one at a time, backward hooks)
        model.zero_grad()
        old_heatmap = old_gc.generate(img_tensor.clone().requires_grad_(True))

        # New method (batched, autograd.grad)
        new_heatmap = new_gc.generate_batch(img_tensor)[0]

        diff = np.max(np.abs(old_heatmap - new_heatmap))
        max_diff = max(max_diff, diff)

    old_gc.remove_hooks()
    new_gc.remove_hook()

    return max_diff


# ── Metric Computation Functions ──────────────────────────────────────────

def get_pixel_ranking(heatmap):
    """
    Stable sort of pixel indices from most to least salient.
    Deterministic tie-breaking by flattened index (lower index first).
    """
    flat = heatmap.flatten()
    # Stable sort: negate values so highest comes first; argsort is stable
    # for equal values, preserving the original index order
    order = np.argsort(-flat, kind='stable')
    return order


def compute_deletion_auc(model, images, heatmaps, device):
    """
    Deletion AUC: progressively mask most-salient pixels with 0.
    Returns per-sample deletion AUC values.
    """
    B = images.shape[0]
    n_pixels = 224 * 224
    results = np.zeros(B)

    # Pre-compute pixel orderings
    orderings = [get_pixel_ranking(heatmaps[i]) for i in range(B)]

    curves = np.zeros((B, len(DELETION_STEPS)))

    for step_idx, frac in enumerate(DELETION_STEPS):
        n_mask = int(frac * n_pixels)

        # Create masked images
        masked = images.clone()
        for i in range(B):
            if n_mask > 0:
                pixels_to_mask = orderings[i][:n_mask]
                rows = pixels_to_mask // 224
                cols = pixels_to_mask % 224
                masked[i, :, rows, cols] = 0.0  # zero in normalised space

        with torch.no_grad():
            logits = model(masked.to(device))
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        curves[:, step_idx] = probs

    # Trapezoid integration over fraction axis [0, 1]
    for i in range(B):
        results[i] = np.trapz(curves[i], DELETION_STEPS)

    return results


def compute_insertion_auc(model, images, heatmaps, device):
    """
    Insertion AUC: start from Gaussian-blurred image, progressively reveal
    most-salient pixels. Higher = more faithful.
    """
    B = images.shape[0]
    n_pixels = 224 * 224
    results = np.zeros(B)

    # Create blurred baseline (per-channel Gaussian blur)
    blurred = images.clone()
    for i in range(B):
        for c in range(3):
            ch = images[i, c].numpy()
            blurred_ch = ndimage.gaussian_filter(ch, sigma=INSERTION_BLUR_SIGMA)
            blurred[i, c] = torch.from_numpy(blurred_ch)

    orderings = [get_pixel_ranking(heatmaps[i]) for i in range(B)]

    curves = np.zeros((B, len(DELETION_STEPS)))

    for step_idx, frac in enumerate(DELETION_STEPS):
        n_reveal = int(frac * n_pixels)

        # Start from blurred, reveal pixels from original
        revealed = blurred.clone()
        for i in range(B):
            if n_reveal > 0:
                pixels_to_reveal = orderings[i][:n_reveal]
                rows = pixels_to_reveal // 224
                cols = pixels_to_reveal % 224
                revealed[i, :, rows, cols] = images[i, :, rows, cols]

        with torch.no_grad():
            logits = model(revealed.to(device))
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        curves[:, step_idx] = probs

    for i in range(B):
        results[i] = np.trapz(curves[i], DELETION_STEPS)

    return results


def compute_entropy(heatmap):
    """Normalised Shannon entropy of the heatmap as a probability distribution."""
    flat = heatmap.flatten()
    total = flat.sum()
    if total <= 0:
        return 1.0  # maximum entropy for all-zero map
    p = flat / total
    p = p[p > 0]  # avoid log(0)
    h = -np.sum(p * np.log(p))
    h_max = np.log(len(flat))
    return h / h_max if h_max > 0 else 0.0


def compute_top_k_mass(heatmap, k_frac=TOP_K_FRACTION):
    """Fraction of total heat in the top k% most-salient pixels."""
    flat = heatmap.flatten()
    total = flat.sum()
    if total <= 0:
        return 0.0
    n_top = max(1, int(k_frac * len(flat)))
    top_vals = np.partition(flat, -n_top)[-n_top:]
    return top_vals.sum() / total


def denormalize_image(img_tensor):
    """Convert normalised tensor (C,H,W) back to uint8 (H,W,C)."""
    img = img_tensor.cpu().numpy()
    mean = IMAGENET_MEAN.reshape(3, 1, 1)
    std = IMAGENET_STD.reshape(3, 1, 1)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def compute_cell_mask(img_tensor):
    """
    Binary mask of the cell vs. black background.
    Threshold: pixel intensity >= CELL_MASK_THRESHOLD on grayscale.
    """
    img_uint8 = denormalize_image(img_tensor)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    mask = (gray >= CELL_MASK_THRESHOLD).astype(np.float32)
    return mask


def compute_on_cell_mass(heatmap, cell_mask):
    """Fraction of heatmap heat falling on the cell (vs. background)."""
    total = heatmap.sum()
    if total <= 0:
        return 0.0
    on_cell = (heatmap * cell_mask).sum()
    return on_cell / total


# ── Control Heatmaps ─────────────────────────────────────────────────────

def generate_random_heatmap(shape=(224, 224), rng_seed=RANDOM_SEED):
    """Random heatmap with fixed RNG seed."""
    rng = np.random.RandomState(rng_seed)
    h = rng.rand(*shape).astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min())
    return h


def generate_centre_prior(shape=(224, 224), sigma=CENTRE_SIGMA):
    """Isotropic Gaussian blob centred at the image centre."""
    cy, cx = shape[0] / 2, shape[1] / 2
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    d = (x - cx) ** 2 + (y - cy) ** 2
    h = np.exp(-d / (2 * sigma ** 2)).astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min())
    return h


# ── Optional Proxy Mask ──────────────────────────────────────────────────

def compute_proxy_parasite_mask(img_tensor):
    """
    Rough parasite pseudo-mask using Giemsa-purple colour thresholding in HSV.
    Returns binary mask (224, 224).
    """
    img_uint8 = denormalize_image(img_tensor)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lower = np.array([PROXY_H_RANGE[0] // 2, PROXY_S_RANGE[0], PROXY_V_RANGE[0]])
    upper = np.array([PROXY_H_RANGE[1] // 2, PROXY_S_RANGE[1], PROXY_V_RANGE[1]])
    mask = cv2.inRange(hsv, lower, upper)
    return (mask > 0).astype(np.float32)


def compute_pointing_game(heatmap, mask):
    """Pointing game: 1 if the max-saliency pixel is inside the mask, else 0."""
    if mask.sum() == 0:
        return np.nan  # no mask region
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(mask[max_idx] > 0)


def compute_top_k_iou(heatmap, mask, k_frac=TOP_K_FRACTION):
    """IoU of top-k% saliency region against the mask."""
    flat = heatmap.flatten()
    n_top = max(1, int(k_frac * len(flat)))
    threshold = np.partition(flat, -n_top)[-n_top]
    saliency_region = (heatmap >= threshold).astype(np.float32)
    intersection = (saliency_region * mask).sum()
    union = ((saliency_region + mask) > 0).sum()
    if union == 0:
        return 0.0
    return intersection / union


# ── Main Pipeline ─────────────────────────────────────────────────────────

def process_one_model_layer(model_type, seed, layer_label, layer_accessor,
                            test_dataset, parasitized_indices, device,
                            max_cells=None):
    """
    Process all parasitized cells for one model/seed/layer combination.
    Returns list of per-cell metric dicts.
    """
    build_fn = get_build_fn(model_type)
    ckpt_pattern = CHECKPOINT_PATTERNS[model_type]
    ckpt_path = os.path.join(SCRIPT_DIR, ckpt_pattern.format(seed=seed))

    if not os.path.isfile(ckpt_path):
        print(f"    [SKIP] Checkpoint not found: {ckpt_path}")
        return []

    model = build_fn().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    target_layer = layer_accessor(model)

    # Get predictions on all parasitized cells to classify TP vs FN
    indices = parasitized_indices
    if max_cells is not None:
        indices = indices[:max_cells]

    # Collect all images + labels
    all_images = []
    all_labels = []
    all_paths = []
    for idx in indices:
        img, label = test_dataset[idx]
        all_images.append(img)
        all_labels.append(label.item())
        all_paths.append(test_dataset.image_paths[idx])

    all_images_tensor = torch.stack(all_images)  # (N, 3, 224, 224)

    # Get predictions in batches (no grad needed)
    all_probs = []
    for start in range(0, len(all_images), DATA_BATCH_SIZE):
        batch = all_images_tensor[start:start + DATA_BATCH_SIZE].to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs)
    all_probs = np.array(all_probs)

    # Pre-generate control heatmaps
    random_heatmap = generate_random_heatmap()
    centre_heatmap = generate_centre_prior()

    # Process in batches for Grad-CAM
    results = []
    n_total = len(indices)

    gradcam = BatchGradCAM(model, target_layer, device)

    for start in range(0, n_total, GRADCAM_BATCH_SIZE):
        end = min(start + GRADCAM_BATCH_SIZE, n_total)
        batch_imgs = all_images_tensor[start:end].to(device)

        # Generate Grad-CAM heatmaps
        try:
            heatmaps = gradcam.generate_batch(batch_imgs)
        except Exception as e:
            print(f"    [ERROR] Grad-CAM failed at batch {start}: {e}")
            # Fill with NaN results
            for i in range(start, end):
                results.append(_empty_result(
                    model_type, seed, layer_label, indices[i],
                    all_paths[i], all_labels[i], all_probs[i]))
            continue

        # CPU images for metric computation
        batch_imgs_cpu = all_images_tensor[start:end]

        # Compute faithfulness metrics
        del_aucs = compute_deletion_auc(model, batch_imgs_cpu, heatmaps, device)
        ins_aucs = compute_insertion_auc(model, batch_imgs_cpu, heatmaps, device)

        # Also compute deletion/insertion for controls (only needs model inference)
        random_hms = np.tile(random_heatmap, (end - start, 1, 1))
        centre_hms = np.tile(centre_heatmap, (end - start, 1, 1))
        del_aucs_random = compute_deletion_auc(model, batch_imgs_cpu, random_hms, device)
        ins_aucs_random = compute_insertion_auc(model, batch_imgs_cpu, random_hms, device)
        del_aucs_centre = compute_deletion_auc(model, batch_imgs_cpu, centre_hms, device)
        ins_aucs_centre = compute_insertion_auc(model, batch_imgs_cpu, centre_hms, device)

        for i in range(end - start):
            cell_idx = start + i
            hm = heatmaps[i]
            img_t = batch_imgs_cpu[i]
            cell_mask = compute_cell_mask(img_t)
            proxy_mask = compute_proxy_parasite_mask(img_t)

            pred_prob = all_probs[cell_idx]
            is_tp = (pred_prob >= THRESHOLD)
            subset = 'TP' if is_tp else 'FN'

            row = {
                'model_type': model_type,
                'seed': seed,
                'layer': layer_label,
                'cell_index': indices[cell_idx],
                'image_path': all_paths[cell_idx],
                'true_label': int(all_labels[cell_idx]),
                'pred_prob': float(pred_prob),
                'subset': subset,
                # Grad-CAM metrics
                'deletion_auc': float(del_aucs[i]),
                'insertion_auc': float(ins_aucs[i]),
                'entropy': float(compute_entropy(hm)),
                'top10_mass': float(compute_top_k_mass(hm)),
                'on_cell_mass': float(compute_on_cell_mass(hm, cell_mask)),
                # Controls
                'deletion_auc_random': float(del_aucs_random[i]),
                'insertion_auc_random': float(ins_aucs_random[i]),
                'entropy_random': float(compute_entropy(random_heatmap)),
                'top10_mass_random': float(compute_top_k_mass(random_heatmap)),
                'on_cell_mass_random': float(compute_on_cell_mass(random_heatmap, cell_mask)),
                'deletion_auc_centre': float(del_aucs_centre[i]),
                'insertion_auc_centre': float(ins_aucs_centre[i]),
                'entropy_centre': float(compute_entropy(centre_heatmap)),
                'top10_mass_centre': float(compute_top_k_mass(centre_heatmap)),
                'on_cell_mass_centre': float(compute_on_cell_mass(centre_heatmap, cell_mask)),
                # Proxy mask (optional tier)
                'proxy_pointing_game': float(compute_pointing_game(hm, proxy_mask)),
                'proxy_iou_top10': float(compute_top_k_iou(hm, proxy_mask)),
                'proxy_mask_area_frac': float(proxy_mask.sum() / (224 * 224)),
            }
            results.append(row)

        if (end % 100 < GRADCAM_BATCH_SIZE) or end == n_total:
            print(f"      Processed {end}/{n_total} cells")

    gradcam.remove_hook()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results


def _empty_result(model_type, seed, layer_label, cell_idx, path, label, prob):
    """Return a result dict filled with NaN for failed computations."""
    return {
        'model_type': model_type, 'seed': seed, 'layer': layer_label,
        'cell_index': cell_idx, 'image_path': path,
        'true_label': int(label), 'pred_prob': float(prob),
        'subset': 'TP' if prob >= THRESHOLD else 'FN',
        'deletion_auc': np.nan, 'insertion_auc': np.nan,
        'entropy': np.nan, 'top10_mass': np.nan, 'on_cell_mass': np.nan,
        'deletion_auc_random': np.nan, 'insertion_auc_random': np.nan,
        'entropy_random': np.nan, 'top10_mass_random': np.nan,
        'on_cell_mass_random': np.nan,
        'deletion_auc_centre': np.nan, 'insertion_auc_centre': np.nan,
        'entropy_centre': np.nan, 'top10_mass_centre': np.nan,
        'on_cell_mass_centre': np.nan,
        'proxy_pointing_game': np.nan, 'proxy_iou_top10': np.nan,
        'proxy_mask_area_frac': np.nan,
    }


def aggregate_per_seed(per_cell_rows):
    """Aggregate per-cell metrics to per-seed means."""
    import pandas as pd
    df = pd.DataFrame(per_cell_rows)

    metric_cols = [
        'deletion_auc', 'insertion_auc', 'entropy', 'top10_mass', 'on_cell_mass',
        'deletion_auc_random', 'insertion_auc_random', 'entropy_random',
        'top10_mass_random', 'on_cell_mass_random',
        'deletion_auc_centre', 'insertion_auc_centre', 'entropy_centre',
        'top10_mass_centre', 'on_cell_mass_centre',
        'proxy_pointing_game', 'proxy_iou_top10', 'proxy_mask_area_frac',
    ]

    # Per seed (all parasitized)
    grouped = df.groupby(['model_type', 'seed', 'layer'])[metric_cols].mean().reset_index()
    grouped['subset'] = 'all_parasitized'
    grouped['n_cells'] = df.groupby(['model_type', 'seed', 'layer']).size().values

    # Per seed, TP subset
    tp_df = df[df['subset'] == 'TP']
    if len(tp_df) > 0:
        tp_grouped = tp_df.groupby(['model_type', 'seed', 'layer'])[metric_cols].mean().reset_index()
        tp_grouped['subset'] = 'TP'
        tp_grouped['n_cells'] = tp_df.groupby(['model_type', 'seed', 'layer']).size().values
        grouped = pd.concat([grouped, tp_grouped], ignore_index=True)

    # Per seed, FN subset
    fn_df = df[df['subset'] == 'FN']
    if len(fn_df) > 0:
        fn_grouped = fn_df.groupby(['model_type', 'seed', 'layer'])[metric_cols].mean().reset_index()
        fn_grouped['subset'] = 'FN'
        fn_grouped['n_cells'] = fn_df.groupby(['model_type', 'seed', 'layer']).size().values
        grouped = pd.concat([grouped, fn_grouped], ignore_index=True)

    return grouped


def aggregate_summary(per_seed_df):
    """Compute mean ± SD across seeds for each model/layer/subset."""
    import pandas as pd

    metric_cols = [
        'deletion_auc', 'insertion_auc', 'entropy', 'top10_mass', 'on_cell_mass',
        'deletion_auc_random', 'insertion_auc_random', 'entropy_random',
        'top10_mass_random', 'on_cell_mass_random',
        'deletion_auc_centre', 'insertion_auc_centre', 'entropy_centre',
        'top10_mass_centre', 'on_cell_mass_centre',
        'proxy_pointing_game', 'proxy_iou_top10', 'proxy_mask_area_frac',
    ]

    rows = []
    for (mt, layer, subset), grp in per_seed_df.groupby(['model_type', 'layer', 'subset']):
        row = {'model_type': mt, 'layer': layer, 'subset': subset,
               'n_seeds': len(grp), 'n_cells_total': int(grp['n_cells'].sum())}
        for col in metric_cols:
            vals = grp[col].dropna().values
            if len(vals) > 0:
                row[f'{col}_mean'] = float(np.mean(vals))
                row[f'{col}_sd'] = float(np.std(vals, ddof=0))
            else:
                row[f'{col}_mean'] = np.nan
                row[f'{col}_sd'] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def run_statistical_tests(per_seed_df):
    """
    Paired t-test and Wilcoxon signed-rank test comparing model/layer
    combinations across seeds. Returns test results as a list of dicts.
    """
    import pandas as pd
    results = []

    # Only compare within 'all_parasitized' subset
    df = per_seed_df[per_seed_df['subset'] == 'all_parasitized'].copy()

    model_layers = df[['model_type', 'layer']].drop_duplicates().values.tolist()

    metrics_to_test = ['deletion_auc', 'insertion_auc', 'entropy', 'top10_mass', 'on_cell_mass']

    for i in range(len(model_layers)):
        for j in range(i + 1, len(model_layers)):
            mt_a, layer_a = model_layers[i]
            mt_b, layer_b = model_layers[j]

            df_a = df[(df['model_type'] == mt_a) & (df['layer'] == layer_a)].sort_values('seed')
            df_b = df[(df['model_type'] == mt_b) & (df['layer'] == layer_b)].sort_values('seed')

            # Match on common seeds
            common_seeds = sorted(set(df_a['seed']) & set(df_b['seed']))
            if len(common_seeds) < 3:
                continue

            df_a = df_a[df_a['seed'].isin(common_seeds)]
            df_b = df_b[df_b['seed'].isin(common_seeds)]

            for metric in metrics_to_test:
                vals_a = df_a[metric].values
                vals_b = df_b[metric].values

                if np.any(np.isnan(vals_a)) or np.any(np.isnan(vals_b)):
                    continue

                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(vals_a, vals_b)

                # Wilcoxon signed-rank test
                try:
                    w_stat, w_pval = stats.wilcoxon(vals_a, vals_b)
                except ValueError:
                    w_stat, w_pval = np.nan, np.nan

                results.append({
                    'comparison': f"{mt_a}/{layer_a} vs {mt_b}/{layer_b}",
                    'metric': metric,
                    'n_seeds': len(common_seeds),
                    'mean_a': float(np.mean(vals_a)),
                    'mean_b': float(np.mean(vals_b)),
                    'diff_mean': float(np.mean(vals_a - vals_b)),
                    't_statistic': float(t_stat),
                    't_pvalue': float(t_pval),
                    'wilcoxon_statistic': float(w_stat) if not np.isnan(w_stat) else '',
                    'wilcoxon_pvalue': float(w_pval) if not np.isnan(w_pval) else '',
                })

    return results


def export_tp_crops(test_dataset, parasitized_indices, model_type, seed,
                    device, n_export=100):
    """Export random TP crops for later manual mask annotation."""
    build_fn = get_build_fn(model_type)
    ckpt_pattern = CHECKPOINT_PATTERNS[model_type]
    ckpt_path = os.path.join(SCRIPT_DIR, ckpt_pattern.format(seed=seed))

    if not os.path.isfile(ckpt_path):
        return

    model = build_fn().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Get predictions
    tp_indices = []
    tp_probs = []
    for idx in parasitized_indices:
        img, label = test_dataset[idx]
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()
        if prob >= THRESHOLD:
            tp_indices.append(idx)
            tp_probs.append(prob)
        if len(tp_indices) >= 500:  # collect enough to sample from
            break

    # Randomly sample n_export
    rng = np.random.RandomState(RANDOM_SEED)
    if len(tp_indices) > n_export:
        chosen = rng.choice(len(tp_indices), n_export, replace=False)
        tp_indices = [tp_indices[c] for c in chosen]
        tp_probs = [tp_probs[c] for c in chosen]

    # Export
    out_dir = os.path.join(OUTPUT_DIR, 'gradcam_proxy_masks',
                           f'{model_type}_seed{seed}')
    os.makedirs(out_dir, exist_ok=True)

    csv_rows = []
    for i, (idx, prob) in enumerate(zip(tp_indices, tp_probs)):
        img, label = test_dataset[idx]
        img_uint8 = denormalize_image(img)
        img_pil = Image.fromarray(img_uint8)
        fname = f"tp_{i:03d}.png"
        img_pil.save(os.path.join(out_dir, fname))
        csv_rows.append({
            'filename': fname,
            'cell_index': idx,
            'image_path': test_dataset.image_paths[idx],
            'pred_prob': prob,
        })

    csv_path = os.path.join(out_dir, 'tp_crops.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'cell_index', 'image_path', 'pred_prob'])
        writer.writeheader()
        writer.writerows(csv_rows)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"    Exported {len(csv_rows)} TP crops to {out_dir}")


def generate_paper_tables(summary_df, stat_results, output_dir):
    """Generate paper_tables.md with compact tables."""
    import pandas as pd

    lines = ["# Grad-CAM Quantitative Metrics\n"]
    lines.append(f"Seeds: {SEEDS} (n = {len(SEEDS)})\n")
    lines.append("All values: mean ± SD across seeds. Unit of analysis = seed.\n")

    # Main metrics table (all_parasitized subset)
    sub = summary_df[summary_df['subset'] == 'all_parasitized'].copy()
    if len(sub) > 0:
        lines.append("## All Parasitized Cells\n")
        lines.append("| Model | Layer | Del. AUC ↓ | Ins. AUC ↑ | Entropy | Top-10% Mass | On-Cell Mass |")
        lines.append("|-------|-------|-----------|-----------|---------|-------------|-------------|")
        for _, row in sub.iterrows():
            label = f"{row['model_type']}"
            layer = f"{row['layer']}"
            del_str = f"{row['deletion_auc_mean']:.3f} ± {row['deletion_auc_sd']:.3f}"
            ins_str = f"{row['insertion_auc_mean']:.3f} ± {row['insertion_auc_sd']:.3f}"
            ent_str = f"{row['entropy_mean']:.3f} ± {row['entropy_sd']:.3f}"
            top_str = f"{row['top10_mass_mean']:.3f} ± {row['top10_mass_sd']:.3f}"
            cell_str = f"{row['on_cell_mass_mean']:.3f} ± {row['on_cell_mass_sd']:.3f}"
            lines.append(f"| {label} | {layer} | {del_str} | {ins_str} | {ent_str} | {top_str} | {cell_str} |")

        # Controls
        lines.append("")
        lines.append("### Controls (mean ± SD across seeds, same cells)\n")
        lines.append("| Control | Del. AUC ↓ | Ins. AUC ↑ | Entropy | Top-10% Mass | On-Cell Mass |")
        lines.append("|---------|-----------|-----------|---------|-------------|-------------|")
        # Use first model's values (controls are identical per cell)
        first = sub.iloc[0]
        for ctrl_name, suffix in [('Random', '_random'), ('Centre-prior', '_centre')]:
            del_str = f"{first[f'deletion_auc{suffix}_mean']:.3f} ± {first[f'deletion_auc{suffix}_sd']:.3f}"
            ins_str = f"{first[f'insertion_auc{suffix}_mean']:.3f} ± {first[f'insertion_auc{suffix}_sd']:.3f}"
            ent_str = f"{first[f'entropy{suffix}_mean']:.3f} ± {first[f'entropy{suffix}_sd']:.3f}"
            top_str = f"{first[f'top10_mass{suffix}_mean']:.3f} ± {first[f'top10_mass{suffix}_sd']:.3f}"
            cell_str = f"{first[f'on_cell_mass{suffix}_mean']:.3f} ± {first[f'on_cell_mass{suffix}_sd']:.3f}"
            lines.append(f"| {ctrl_name} | {del_str} | {ins_str} | {ent_str} | {top_str} | {cell_str} |")

    lines.append("")

    # TP and FN subsets
    for subset_name in ['TP', 'FN']:
        sub = summary_df[summary_df['subset'] == subset_name].copy()
        if len(sub) > 0:
            lines.append(f"## {subset_name} Subset\n")
            lines.append("| Model | Layer | Del. AUC ↓ | Ins. AUC ↑ | Entropy | Top-10% Mass | On-Cell Mass | n_seeds |")
            lines.append("|-------|-------|-----------|-----------|---------|-------------|-------------|---------|")
            for _, row in sub.iterrows():
                del_str = f"{row['deletion_auc_mean']:.3f} ± {row['deletion_auc_sd']:.3f}"
                ins_str = f"{row['insertion_auc_mean']:.3f} ± {row['insertion_auc_sd']:.3f}"
                ent_str = f"{row['entropy_mean']:.3f} ± {row['entropy_sd']:.3f}"
                top_str = f"{row['top10_mass_mean']:.3f} ± {row['top10_mass_sd']:.3f}"
                cell_str = f"{row['on_cell_mass_mean']:.3f} ± {row['on_cell_mass_sd']:.3f}"
                lines.append(f"| {row['model_type']} | {row['layer']} | {del_str} | {ins_str} | {ent_str} | {top_str} | {cell_str} | {row['n_seeds']} |")
            lines.append("")

    # Optional proxy metrics
    sub = summary_df[summary_df['subset'] == 'all_parasitized'].copy()
    if len(sub) > 0 and not sub['proxy_pointing_game_mean'].isna().all():
        lines.append("## PROXY: Giemsa-Purple Pseudo-Mask Metrics (not ground truth)\n")
        lines.append("> These use automated colour thresholding, not manual annotations.\n")
        lines.append("| Model | Layer | Pointing Game | IoU (top-10%) | Mask Area Frac |")
        lines.append("|-------|-------|--------------|---------------|----------------|")
        for _, row in sub.iterrows():
            pg = f"{row['proxy_pointing_game_mean']:.3f} ± {row['proxy_pointing_game_sd']:.3f}"
            iou = f"{row['proxy_iou_top10_mean']:.3f} ± {row['proxy_iou_top10_sd']:.3f}"
            area = f"{row['proxy_mask_area_frac_mean']:.3f} ± {row['proxy_mask_area_frac_sd']:.3f}"
            lines.append(f"| {row['model_type']} | {row['layer']} | {pg} | {iou} | {area} |")
        lines.append("")

    # Statistical tests
    if stat_results:
        lines.append("## Statistical Comparisons (paired across seeds)\n")
        lines.append("| Comparison | Metric | n | Diff (mean) | t-stat | t p-value | W-stat | W p-value |")
        lines.append("|-----------|--------|---|-------------|--------|-----------|--------|-----------|")
        for r in stat_results:
            w_s = f"{r['wilcoxon_statistic']}" if r['wilcoxon_statistic'] != '' else 'N/A'
            w_p = f"{r['wilcoxon_pvalue']:.4f}" if r['wilcoxon_pvalue'] != '' else 'N/A'
            lines.append(
                f"| {r['comparison']} | {r['metric']} | {r['n_seeds']} | "
                f"{r['diff_mean']:.4f} | {r['t_statistic']:.3f} | {r['t_pvalue']:.4f} | "
                f"{w_s} | {w_p} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Quantitative Grad-CAM Evaluation')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Max parasitized cells per model/seed (for dev runs)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nRNG seeds used: {RANDOM_SEED} (controls, sampling)")
    print(f"Analysis parameters:")
    print(f"  Deletion/Insertion steps: {len(DELETION_STEPS)} (0% to 100% in 5% steps)")
    print(f"  Insertion blur sigma: {INSERTION_BLUR_SIGMA}")
    print(f"  Cell mask threshold: {CELL_MASK_THRESHOLD}")
    print(f"  Top-k fraction: {TOP_K_FRACTION}")
    print(f"  Centre-prior sigma: {CENTRE_SIGMA}")
    print(f"  Grad-CAM batch size: {GRADCAM_BATCH_SIZE}")
    print(f"  Proxy mask HSV: H={PROXY_H_RANGE}, S={PROXY_S_RANGE}, V={PROXY_V_RANGE}")

    # Load test data
    print("\nLoading test dataset...")
    test_dataset, test_idx, full_dataset = get_test_data()

    # Identify all parasitized cells
    parasitized_indices = [i for i in range(len(test_dataset))
                           if test_dataset.labels[i].item() == 1]
    print(f"Parasitized cells in test set: {len(parasitized_indices)}")

    if args.max_cells:
        print(f"  --max-cells flag: using at most {args.max_cells} cells per model/seed")

    # ── Verification step ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Verifying batched Grad-CAM against existing gradcam.py")
    print("=" * 60)

    # Verify on hybrid_cbam seed 42, pre-CBAM layer (matches existing default)
    ckpt_42 = os.path.join(SCRIPT_DIR, 'best_model_seed42.pt')
    if os.path.isfile(ckpt_42):
        build_fn = get_build_fn('hybrid_cbam')
        model_verify = build_fn().to(device)
        state = torch.load(ckpt_42, map_location=device, weights_only=True)
        model_verify.load_state_dict(state)

        target_layer = model_verify.features[7][0]  # pre-CBAM, matches gradcam.py default
        max_diff = verify_against_existing_gradcam(
            model_verify, target_layer, test_dataset, device, n_check=10)
        print(f"  Max abs diff (10 cells, normalised): {max_diff:.6f}")
        if max_diff < 1e-4:
            print("  ✓ PASS: Batched implementation reproduces existing gradcam.py")
        else:
            print(f"  ✗ FAIL: Difference {max_diff:.6f} exceeds 1e-4 tolerance!")
            print("  Continuing anyway — results may differ from existing heatmaps.")

        del model_verify
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        print("  [SKIP] Hybrid CBAM seed 42 checkpoint not found for verification")

    # ── Benchmarking ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Benchmarking: timing one model/layer on 50 cells")
    print("=" * 60)

    # Find first available config for benchmarking
    bench_done = False
    for model_type, layer_label, layer_accessor in LAYER_CONFIGS:
        for seed in SEEDS:
            ckpt = os.path.join(SCRIPT_DIR,
                                CHECKPOINT_PATTERNS[model_type].format(seed=seed))
            if os.path.isfile(ckpt):
                t0 = time.time()
                _ = process_one_model_layer(
                    model_type, seed, layer_label, layer_accessor,
                    test_dataset, parasitized_indices, device, max_cells=50)
                bench_time = time.time() - t0
                n_cells = min(50, len(parasitized_indices))
                cells_to_do = args.max_cells if args.max_cells else len(parasitized_indices)
                n_configs = 0
                for mt, _, _ in LAYER_CONFIGS:
                    for s in SEEDS:
                        cp = os.path.join(SCRIPT_DIR,
                                          CHECKPOINT_PATTERNS[mt].format(seed=s))
                        if os.path.isfile(cp):
                            n_configs += 1
                est_total = (bench_time / n_cells) * cells_to_do * n_configs
                print(f"  Benchmark: {bench_time:.1f}s for {n_cells} cells")
                print(f"  Estimated total: {est_total / 60:.1f} min "
                      f"({n_configs} configs × {cells_to_do} cells)")
                bench_done = True
                break
        if bench_done:
            break

    if not bench_done:
        print("  [SKIP] No checkpoints found for benchmarking")

    # ── Full Run ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: Full Grad-CAM Metrics Computation")
    print("=" * 60)

    all_cell_results = []

    for model_type, layer_label, layer_accessor in LAYER_CONFIGS:
        for seed in SEEDS:
            print(f"\n  Processing: {model_type} | seed {seed} | {layer_label}")
            results = process_one_model_layer(
                model_type, seed, layer_label, layer_accessor,
                test_dataset, parasitized_indices, device,
                max_cells=args.max_cells)
            all_cell_results.extend(results)
            print(f"    → {len(results)} cells processed")

    if not all_cell_results:
        print("\n[ERROR] No results generated. Check checkpoints.")
        return

    # Save per-cell CSV
    import pandas as pd
    cell_df = pd.DataFrame(all_cell_results)
    cell_csv = os.path.join(OUTPUT_DIR, 'gradcam_metrics_per_cell.csv')
    cell_df.to_csv(cell_csv, index=False)
    print(f"\nSaved: {cell_csv} ({len(cell_df)} rows)")

    # Aggregate per seed
    per_seed_df = aggregate_per_seed(all_cell_results)
    seed_csv = os.path.join(OUTPUT_DIR, 'gradcam_metrics_per_seed.csv')
    per_seed_df.to_csv(seed_csv, index=False)
    print(f"Saved: {seed_csv} ({len(per_seed_df)} rows)")

    # Aggregate summary (mean ± SD across seeds)
    summary_df = aggregate_summary(per_seed_df)
    summary_csv = os.path.join(OUTPUT_DIR, 'gradcam_metrics_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv} ({len(summary_df)} rows)")

    # Statistical tests
    print("\n  Running statistical comparisons...")
    stat_results = run_statistical_tests(per_seed_df)

    # Save stat test results
    if stat_results:
        stat_csv = os.path.join(OUTPUT_DIR, 'gradcam_stat_tests.csv')
        pd.DataFrame(stat_results).to_csv(stat_csv, index=False)
        print(f"Saved: {stat_csv}")

    # Generate paper tables
    paper_text = generate_paper_tables(summary_df, stat_results, OUTPUT_DIR)

    # Export TP crops for annotation (one seed per model, using first available)
    print("\n  Exporting TP crops for manual annotation...")
    for model_type in ['hybrid_cbam', 'hybrid_eca', 'baseline']:
        for seed in SEEDS:
            ckpt = os.path.join(SCRIPT_DIR,
                                CHECKPOINT_PATTERNS[model_type].format(seed=seed))
            if os.path.isfile(ckpt):
                export_tp_crops(test_dataset, parasitized_indices,
                                model_type, seed, device, n_export=100)
                break  # one seed per model

    # Return paper text for merging later
    return paper_text, summary_df, stat_results


if __name__ == '__main__':
    result = main()
    if result:
        paper_text, _, _ = result
        # Will be merged with BBBC041 tables in a separate step
        paper_path = os.path.join(OUTPUT_DIR, 'gradcam_paper_tables.md')
        with open(paper_path, 'w', encoding='utf-8') as f:
            f.write(paper_text)
        print(f"\nSaved: {paper_path}")
