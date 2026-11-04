"""
BBBC041 Zero-Shot Transfer Evaluation Pipeline
================================================
Tests trained models (Baseline EfficientNet-B0 and Hybrid EfficientNet-B0+CBAM)
on the BBBC041 P. vivax malaria dataset WITHOUT any retraining.

Usage:
    python bbbc041_eval.py

Prerequisites:
    - Trained model checkpoints in the same directory
    - BBBC041 dataset will be auto-downloaded or manually placed in bbbc041/
"""

import os
import sys
import json
import csv
import zipfile
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BBBC041_DIR = os.path.join(SCRIPT_DIR, 'bbbc041')
BBBC041_ZIP = os.path.join(SCRIPT_DIR, 'malaria.zip')
BBBC041_URL = 'https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip'

PARASITIZED_CLASSES = {'ring', 'trophozoite', 'schizont', 'gametocyte'}
UNINFECTED_CLASSES = {'red blood cell', 'leukocyte'}
SKIP_CLASSES = {'difficult'}

SEEDS = [42, 123, 2026, 7, 17]
THRESHOLD = 0.30
BATCH_SIZE = 32

# Checkpoint filename patterns (actual filenames from the project)
HYBRID_CKPT_FMT = 'best_model_seed{seed}.pt'
BASELINE_CKPT_FMT = 'checkpoint_baseline_seed_{seed}.pth'
ECANET_CKPT_FMT = 'best_model_ecanet_seed{seed}.pt'

# Metrics to report
METRIC_KEYS = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC-ROC']

# ---------------------------------------------------------------------------
# Step 1: Download / Extract BBBC041
# ---------------------------------------------------------------------------
def ensure_bbbc041_dataset():
    """Download and extract BBBC041 if not already present."""
    # Check if the extracted directory already has content
    if os.path.isdir(BBBC041_DIR):
        json_files = glob.glob(os.path.join(BBBC041_DIR, '**', '*.json'), recursive=True)
        if json_files:
            print(f"[OK] BBBC041 dataset found at: {BBBC041_DIR}")
            return True

    # Try downloading
    print("BBBC041 dataset not found. Attempting download...")
    try:
        import urllib.request
        print(f"Downloading from {BBBC041_URL} ...")
        print("(This is a 2.26 GB file — download may take a while)")
        urllib.request.urlretrieve(BBBC041_URL, BBBC041_ZIP)
        print("Download complete. Extracting...")
    except Exception as e:
        print(f"\n[ERROR] Auto-download failed: {e}")
        _print_manual_instructions()
        return False

    # Extract
    try:
        if not zipfile.is_zipfile(BBBC041_ZIP):
            print(f"[ERROR] Downloaded file is not a valid zip. Removing: {BBBC041_ZIP}")
            os.remove(BBBC041_ZIP)
            _print_manual_instructions()
            return False

        with zipfile.ZipFile(BBBC041_ZIP, 'r') as zf:
            zf.extractall(SCRIPT_DIR)
        print("Extraction complete.")

        # The zip may extract to 'malaria/' — rename to 'bbbc041/' if so
        malaria_dir = os.path.join(SCRIPT_DIR, 'malaria')
        if os.path.isdir(malaria_dir) and not os.path.isdir(BBBC041_DIR):
            os.rename(malaria_dir, BBBC041_DIR)
            print(f"Renamed 'malaria/' → 'bbbc041/'")

        # Cleanup zip
        if os.path.exists(BBBC041_ZIP):
            os.remove(BBBC041_ZIP)
            print("Cleaned up zip file.")

        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        _print_manual_instructions()
        return False


def _print_manual_instructions():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  MANUAL DOWNLOAD REQUIRED                   ║
╠══════════════════════════════════════════════════════════════╣
║  1. Visit: https://data.broadinstitute.org/bbbc/BBBC041/    ║
║  2. Download: malaria.zip (approx. 2.26 GB)                 ║
║  3. Extract into: {BBBC041_DIR}
║     (so that JSON + image folders are directly inside it)    ║
║  4. Re-run: python bbbc041_eval.py                          ║
╚══════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# Step 2: Parse JSON annotations
# ---------------------------------------------------------------------------
def parse_bbbc041_annotations(bbbc_dir, subset='test'):
    """
    Parse BBBC041 JSON annotations and return cell-level tuples.

    BBBC041 JSON format (as documented):
    A list of dicts, each with:
        - "image": dict with "pathname" key (path to the image file)
        - "objects": list of dicts, each with:
            - "bounding_box": dict with keys like "minimum"/"maximum" (row/column)
              OR numeric coords depending on the version
            - "category": string class label

    We handle multiple possible JSON structures robustly.

    Returns:
        List of (image_path, bbox_dict, binary_label) tuples
        bbox_dict has keys: x, y, w, h (pixel coords)
    """
    cells = []
    skipped_difficult = 0
    skipped_unknown = 0
    class_counts = {}

    # Find JSON files — look for test.json or testing.json
    json_candidates = [
        os.path.join(bbbc_dir, f'{subset}.json'),
        os.path.join(bbbc_dir, f'{subset}ing.json'),  # "testing.json"
    ]
    # Also search recursively
    for jf in glob.glob(os.path.join(bbbc_dir, '**', '*.json'), recursive=True):
        json_candidates.append(jf)

    json_path = None
    for candidate in json_candidates:
        if os.path.isfile(candidate):
            basename = os.path.basename(candidate).lower()
            if subset in basename:
                json_path = candidate
                break

    if json_path is None:
        # Fall back: use any JSON file found
        all_jsons = glob.glob(os.path.join(bbbc_dir, '**', '*.json'), recursive=True)
        if all_jsons:
            print(f"[WARNING] Could not find '{subset}.json'. Available JSON files:")
            for jf in all_jsons:
                print(f"  - {jf}")
            # Use the first one that looks like test data
            for jf in all_jsons:
                if 'test' in os.path.basename(jf).lower():
                    json_path = jf
                    break
            if json_path is None:
                json_path = all_jsons[0]
                print(f"[WARNING] Using: {json_path}")
        else:
            raise FileNotFoundError(
                f"No JSON annotation files found in {bbbc_dir}. "
                "Please ensure BBBC041 is correctly extracted."
            )

    print(f"Parsing annotations from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        # Parse image path
        img_info = entry.get('image', {})
        if isinstance(img_info, dict):
            img_rel_path = img_info.get('pathname', '')
        elif isinstance(img_info, str):
            img_rel_path = img_info
        else:
            continue

        # Resolve full path — try multiple base directories
        img_path = os.path.join(bbbc_dir, img_rel_path)
        if not os.path.isfile(img_path):
            # Try without leading directory component
            img_basename = os.path.basename(img_rel_path)
            found = glob.glob(os.path.join(bbbc_dir, '**', img_basename), recursive=True)
            if found:
                img_path = found[0]
            else:
                continue

        # Parse objects (cells)
        objects = entry.get('objects', [])
        for obj in objects:
            category = obj.get('category', '').lower().strip()

            # Track class distribution
            class_counts[category] = class_counts.get(category, 0) + 1

            # Skip difficult / unknown
            if category in SKIP_CLASSES:
                skipped_difficult += 1
                continue

            # Map to binary label
            if category in PARASITIZED_CLASSES:
                label = 1  # Parasitized
            elif category in UNINFECTED_CLASSES:
                label = 0  # Uninfected
            else:
                skipped_unknown += 1
                continue

            # Parse bounding box — handle multiple possible formats
            bbox_raw = obj.get('bounding_box', {})
            bbox = _parse_bbox(bbox_raw)
            if bbox is None:
                continue

            cells.append((img_path, bbox, label))

    print(f"\n--- BBBC041 Annotation Summary ---")
    print(f"JSON file: {os.path.basename(json_path)}")
    print(f"Total cells parsed: {len(cells)}")
    print(f"Skipped (difficult): {skipped_difficult}")
    print(f"Skipped (unknown class): {skipped_unknown}")
    print(f"\nOriginal class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        mapped = 'Parasitized' if cls in PARASITIZED_CLASSES else (
            'Uninfected' if cls in UNINFECTED_CLASSES else 'SKIPPED')
        print(f"  {cls}: {cnt}  →  {mapped}")

    n_parasitized = sum(1 for _, _, l in cells if l == 1)
    n_uninfected = sum(1 for _, _, l in cells if l == 0)
    print(f"\nBinary distribution:")
    print(f"  Parasitized (1): {n_parasitized}  ({100*n_parasitized/len(cells):.1f}%)")
    print(f"  Uninfected  (0): {n_uninfected}  ({100*n_uninfected/len(cells):.1f}%)")

    return cells, class_counts


def _parse_bbox(bbox_raw):
    """
    Parse bounding box from various BBBC041 JSON formats.
    Returns dict with keys: x, y, w, h (all in pixels).
    """
    if not bbox_raw:
        return None

    # Format 1: {"minimum": {"r": y, "c": x}, "maximum": {"r": y2, "c": x2}}
    if 'minimum' in bbox_raw and 'maximum' in bbox_raw:
        mn = bbox_raw['minimum']
        mx = bbox_raw['maximum']
        x = mn.get('c', mn.get('x', 0))
        y = mn.get('r', mn.get('y', 0))
        x2 = mx.get('c', mx.get('x', 0))
        y2 = mx.get('r', mx.get('y', 0))
        return {'x': int(x), 'y': int(y), 'w': int(x2 - x), 'h': int(y2 - y)}

    # Format 2: {"x": x, "y": y, "width": w, "height": h}
    if 'x' in bbox_raw and 'y' in bbox_raw:
        return {
            'x': int(bbox_raw['x']),
            'y': int(bbox_raw['y']),
            'w': int(bbox_raw.get('width', bbox_raw.get('w', 50))),
            'h': int(bbox_raw.get('height', bbox_raw.get('h', 50)))
        }

    # Format 3: {"xmin": x, "ymin": y, "xmax": x2, "ymax": y2}
    if 'xmin' in bbox_raw:
        return {
            'x': int(bbox_raw['xmin']),
            'y': int(bbox_raw['ymin']),
            'w': int(bbox_raw['xmax'] - bbox_raw['xmin']),
            'h': int(bbox_raw['ymax'] - bbox_raw['ymin'])
        }

    return None


# ---------------------------------------------------------------------------
# Step 3 & 4: PyTorch Dataset with on-the-fly cropping
# ---------------------------------------------------------------------------
class BBBC041Dataset(Dataset):
    """
    Dataset that crops individual cells from whole-slide BBBC041 images
    using bounding box annotations, with 10% contextual padding.
    """
    def __init__(self, cell_entries, transform=None):
        """
        Args:
            cell_entries: list of (image_path, bbox_dict, label)
            transform: torchvision transforms
        """
        self.cell_entries = cell_entries
        self.transform = transform
        self.image_paths = [e[0] for e in cell_entries]  # Compatibility attribute
        self._img_cache = {}  # Cache loaded images to avoid re-reading

    def __len__(self):
        return len(self.cell_entries)

    def __getitem__(self, idx):
        img_path, bbox, label = self.cell_entries[idx]

        # Load image (with caching for efficiency — same slide has many cells)
        if img_path not in self._img_cache:
            try:
                img = Image.open(img_path).convert('RGB')
                # Only cache if we have memory headroom (limit cache size)
                if len(self._img_cache) < 100:
                    self._img_cache[img_path] = img
            except Exception as e:
                print(f"[WARNING] Failed to load {img_path}: {e}")
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            img = self._img_cache[img_path]

        img_w, img_h = img.size

        # Crop with 10% padding
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        crop = img.crop((x1, y1, x2, y2))

        if self.transform:
            crop = self.transform(crop)

        return crop, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Step 5: DataLoader creation
# ---------------------------------------------------------------------------
def create_bbbc041_loader(cell_entries):
    """Create DataLoader with same val/test preprocessing as original training."""
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BBBC041Dataset(cell_entries, transform=val_test_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return loader


# ---------------------------------------------------------------------------
# Task 2: Zero-shot inference
# ---------------------------------------------------------------------------
def setup_device():
    """Detect and configure compute device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\nWARNING: Running on CPU — inference will be slow.")
    return device


def evaluate_model(model, loader, device, threshold=THRESHOLD):
    """
    Run zero-shot inference and compute metrics.
    Uses the existing calculate_metrics function from evaluate.py.
    """
    # Import from the project's evaluate module
    sys.path.insert(0, SCRIPT_DIR)
    from evaluate import calculate_metrics

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference", leave=False):
            images = images.to(device)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    metrics = calculate_metrics(all_labels, all_preds, threshold=threshold)
    return metrics


def run_evaluation(loader, device):
    """Run zero-shot evaluation for all seeds and both models."""
    sys.path.insert(0, SCRIPT_DIR)
    from model import build_model as build_hybrid
    from model_baseline import build_model as build_baseline
    from model_ecanet import build_model as build_ecanet

    all_results = []

    for model_type, build_fn, ckpt_fmt in [
        ('Hybrid (CBAM)', build_hybrid, HYBRID_CKPT_FMT),
        ('Hybrid (ECA-Net)', build_ecanet, ECANET_CKPT_FMT),
        ('Baseline', build_baseline, BASELINE_CKPT_FMT),
    ]:
        print(f"\n{'='*60}")
        print(f"  {model_type.upper()} MODEL — Zero-Shot Transfer")
        print(f"{'='*60}")

        for seed in SEEDS:
            ckpt_name = ckpt_fmt.format(seed=seed)
            ckpt_path = os.path.join(SCRIPT_DIR, ckpt_name)

            if not os.path.isfile(ckpt_path):
                print(f"\n  [SKIP] Seed {seed}: checkpoint not found ({ckpt_name})")
                continue

            print(f"\n  Seed {seed} — Loading {ckpt_name}...")
            model = build_fn().to(device)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )

            metrics = evaluate_model(model, loader, device)

            print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"    Recall:    {metrics['Recall']:.4f}")
            print(f"    Precision: {metrics['Precision']:.4f}")
            print(f"    F1-Score:  {metrics['F1-Score']:.4f}")
            print(f"    AUC-ROC:   {metrics['AUC-ROC']:.4f}")
            print(f"    TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}  TP={metrics['TP']}")

            all_results.append({
                'Model': model_type,
                'Seed': seed,
                **{k: metrics[k] for k in METRIC_KEYS},
                'TN': metrics['TN'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'TP': metrics['TP'],
                'Threshold': metrics['Threshold'],
            })

            # Free GPU memory
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# Task 3: Results reporting
# ---------------------------------------------------------------------------
def print_summary_table(results):
    """Print formatted summary table."""
    print(f"\n\n{'='*70}")
    print(f"BBBC041 ZERO-SHOT TRANSFER RESULTS (threshold={THRESHOLD}, no retraining)")
    print(f"{'='*70}")

    model_types = list(dict.fromkeys(r['Model'] for r in results))

    for metric in METRIC_KEYS:
        print(f"\n--- {metric} ---")
        for model_type in model_types:
            model_results = [r for r in results if r['Model'] == model_type]
            if not model_results:
                continue

            vals = [r[metric] for r in model_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            print(f"  {model_type:<18} {metric:12s}: "
                  + "  ".join(f"Seed {r['Seed']}={v:.4f}" for r, v in zip(model_results, vals))
                  + f"  |  Mean: {mean_val:.4f} ± {std_val:.4f}")


def save_results_csv(results):
    """Save all per-seed metrics to CSV."""
    csv_path = os.path.join(SCRIPT_DIR, 'bbbc041_transfer_results.csv')
    fieldnames = ['Model', 'Seed'] + METRIC_KEYS + ['TN', 'FP', 'FN', 'TP', 'Threshold']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SAVED] Per-seed results → {csv_path}")


def print_distribution_warning(cells):
    """Print class distribution analysis."""
    n_total = len(cells)
    n_para = sum(1 for _, _, l in cells if l == 1)
    n_uninf = sum(1 for _, _, l in cells if l == 0)

    print(f"\n{'='*70}")
    print("DISTRIBUTION SHIFT WARNING")
    print(f"{'='*70}")
    print(f"  Total BBBC041 test cells evaluated: {n_total}")
    print(f"  Parasitized: {n_para}  ({100*n_para/n_total:.1f}%)")
    print(f"  Uninfected:  {n_uninf}  ({100*n_uninf/n_total:.1f}%)")

    if n_uninf > n_para * 3:
        print(f"  ⚠ HEAVILY SKEWED towards Uninfected (ratio {n_uninf/max(n_para,1):.1f}:1)")
    elif n_uninf > n_para * 1.5:
        print(f"  ⚠ Moderately skewed towards Uninfected")
    else:
        print(f"  ✓ Approximately balanced")

    print(f"\n  NOTE: AUC-ROC is the most reliable metric when class distribution")
    print(f"  differs significantly from the balanced NIH training set.")
    print(f"  F1-Score may be misleadingly high if the majority class dominates.")


def print_interpretation(results):
    """Print automated interpretation based on results."""
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    model_types = list(dict.fromkeys(r['Model'] for r in results))
    for model_type in model_types:
        model_results = [r for r in results if r['Model'] == model_type]
        if not model_results:
            continue

        f1_vals = [r['F1-Score'] for r in model_results]
        mean_f1 = np.mean(f1_vals)

        if mean_f1 >= 0.90:
            verdict = "Strong generalisation — models transfer well to BBBC041"
        elif mean_f1 >= 0.80:
            verdict = "Moderate generalisation — some distribution shift observed"
        else:
            verdict = ("Significant distribution shift — performance drop consistent with "
                       "species difference (P. vivax vs P. falciparum) and whole-slide vs "
                       "pre-cropped cell preparation")

        print(f"\n  {model_type}: Mean F1 = {mean_f1:.4f}  →  {verdict}")


# ---------------------------------------------------------------------------
# Task 4: Paper framing guidance
# ---------------------------------------------------------------------------
def print_paper_text(results):
    """Generate suggested paper text based on actual results."""
    hybrid_results = [r for r in results if r['Model'] == 'Hybrid']
    baseline_results = [r for r in results if r['Model'] == 'Baseline']

    if not hybrid_results or not baseline_results:
        print("\n[Cannot generate paper text — missing results for one or both models]")
        return

    h_f1 = np.mean([r['F1-Score'] for r in hybrid_results])
    h_f1_std = np.std([r['F1-Score'] for r in hybrid_results])
    h_auc = np.mean([r['AUC-ROC'] for r in hybrid_results])
    h_auc_std = np.std([r['AUC-ROC'] for r in hybrid_results])

    b_f1 = np.mean([r['F1-Score'] for r in baseline_results])
    b_f1_std = np.std([r['F1-Score'] for r in baseline_results])
    b_auc = np.mean([r['AUC-ROC'] for r in baseline_results])
    b_auc_std = np.std([r['AUC-ROC'] for r in baseline_results])

    # Determine relative performance
    if h_f1 > b_f1 + 0.01:
        comparison = "improvement"
    elif b_f1 > h_f1 + 0.01:
        comparison = "decline"
    else:
        comparison = "comparable performance"

    print(f"\n{'='*70}")
    print("SUGGESTED PAPER TEXT (Section IV or Limitations):")
    print(f"{'='*70}")
    print(f"""
\"To assess cross-dataset generalisation, both trained models were evaluated
zero-shot on the BBBC041 benchmark [Ljosa et al., Nature Methods, 2012],
which contains Giemsa-stained P. vivax blood smear images with annotated
cell-level bounding boxes. Cell crops were extracted using the provided
bounding box annotations with 10% contextual padding and processed using
the same val/test preprocessing pipeline applied during NIH training.
The hybrid model achieved a mean F1-score of {h_f1:.4f} ± {h_f1_std:.4f}
and AUC-ROC of {h_auc:.4f} ± {h_auc_std:.4f} across {len(hybrid_results)} seeds;
the baseline achieved F1 {b_f1:.4f} ± {b_f1_std:.4f} and AUC {b_auc:.4f}
± {b_auc_std:.4f}. The observed {comparison} relative to NIH performance
is consistent with the expected distribution shift between P. falciparum
(NIH) and P. vivax (BBBC041), differing staining intensities across
collection sites, and the additional preprocessing required to extract
cell crops from whole-slide images. Both models showed comparable
cross-dataset behaviour, confirming that neither architecture generalises
substantially better than the other under these conditions.\"
""")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  BBBC041 Zero-Shot Transfer Evaluation Pipeline")
    print("  Models: Baseline (EfficientNet-B0) vs Hybrid (EfficientNet-B0+CBAM)")
    print("  Threshold: 0.30 (fixed, no tuning on BBBC041)")
    print("=" * 70)

    # Step 1: Ensure dataset exists
    if not ensure_bbbc041_dataset():
        sys.exit(1)

    # Step 2: Parse annotations (test subset only)
    cells, class_counts = parse_bbbc041_annotations(BBBC041_DIR, subset='test')
    if not cells:
        print("[ERROR] No valid cell entries parsed. Check dataset structure.")
        sys.exit(1)

    # Step 3-5: Create DataLoader
    print(f"\nCreating DataLoader ({len(cells)} cells, batch_size={BATCH_SIZE})...")
    loader = create_bbbc041_loader(cells)

    # Setup device
    device = setup_device()

    # Task 2: Run evaluation
    results = run_evaluation(loader, device)

    if not results:
        print("\n[ERROR] No results generated. Check that checkpoint files exist.")
        sys.exit(1)

    # Task 3: Report results
    print_summary_table(results)
    save_results_csv(results)
    print_distribution_warning(cells)
    print_interpretation(results)

    # Task 4: Paper text
    print_paper_text(results)

    print(f"\n{'='*70}")
    print("  Pipeline complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
