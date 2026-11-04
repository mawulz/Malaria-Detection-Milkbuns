import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Inline display support for Kaggle / Jupyter notebooks
try:
    from IPython.display import display as ipy_display, Image as IPImage
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = (
            target_layer if target_layer is not None
            else model.features[7][0]
        )
        self.activations = None
        self.gradients = None
        self.hooks = []

        self.hooks.append(
            self.target_layer.register_forward_hook(self.save_activation)
        )
        self.hooks.append(
            self.target_layer.register_full_backward_hook(self.save_gradient)
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, 0]
        score.backward()

        if self.activations is None or self.gradients is None:
            print("  [GradCAM] Warning: hooks did not fire — returning blank heatmap.")
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        acts  = self.activations[0]
        grads = self.gradients[0]

        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        if np.max(cam) == 0:
            print("  [GradCAM] Warning: heatmap is all zeros — gradients may not "
                  "be flowing through the target layer.")
        else:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def overlay_heatmap(image_tensor, heatmap):
    """Denormalise image tensor and blend with jet-colourmap heatmap."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = image_tensor.cpu().detach().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)

    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    overlay = 0.6 * img + 0.4 * heatmap_colored
    overlay = overlay.astype(np.uint8)

    return Image.fromarray(overlay)


def _add_label(img: Image.Image, text: str) -> Image.Image:
    """Draw a small label banner below the image."""
    label_h = 20
    banner = Image.new('RGB', (img.width, label_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(banner)
    draw.text((4, 3), text, fill=(220, 220, 220))
    combined = Image.new('RGB', (img.width, img.height + label_h))
    combined.paste(img, (0, 0))
    combined.paste(banner, (0, img.height))
    return combined


def visualise_false_negatives(
    model, test_loader, device,
    output_dir='gradcam_fn', max_samples=60, threshold=0.30
):
    os.makedirs(output_dir, exist_ok=True)
    gradcam = GradCAM(model)

    fn_count = 0
    results   = []

    # Handle both Subset and direct MalariaDataset
    dataset      = test_loader.dataset
    base_dataset = (
        dataset.dataset
        if isinstance(dataset, torch.utils.data.Subset)
        else dataset
    )

    model.eval()

    with tqdm(total=max_samples, desc=f"GradCAM FNs → {output_dir}") as pbar:
        global_idx = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Batch inference — no grad needed here
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    probs   = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()

            for i in range(len(images)):
                true_label = int(labels[i].item())
                pred_prob  = float(probs[i])

                if true_label == 1 and pred_prob < threshold:
                    # Map back to image file path
                    abs_idx  = (
                        dataset.indices[global_idx]
                        if isinstance(dataset, torch.utils.data.Subset)
                        else global_idx
                    )
                    filepath = base_dataset.image_paths[abs_idx]
                    filename = os.path.basename(filepath)

                    # Generate Grad-CAM with gradient-enabled forward pass
                    input_tensor = images[i].unsqueeze(0).clone()
                    input_tensor.requires_grad_(True)

                    try:
                        heatmap     = gradcam.generate(input_tensor)
                        overlay_img = overlay_heatmap(images[i], heatmap)
                    except Exception as e:
                        print(f"  [GradCAM] Skipped sample {fn_count}: {e}")
                        global_idx += 1
                        continue

                    # Reconstruct denormalised original
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    raw  = images[i].cpu().detach().numpy()
                    raw  = std * raw + mean
                    raw  = np.clip(raw, 0, 1)
                    raw  = np.transpose(raw, (1, 2, 0))
                    raw  = (raw * 255).astype(np.uint8)
                    orig_img = Image.fromarray(raw)

                    # Add labels under each panel
                    orig_labelled    = _add_label(orig_img,    "Original")
                    overlay_labelled = _add_label(overlay_img, "Grad-CAM")

                    # Side-by-side composite
                    w, h = orig_labelled.size
                    composite = Image.new('RGB', (w * 2, h))
                    composite.paste(orig_labelled,    (0, 0))
                    composite.paste(overlay_labelled, (w, 0))

                    # Save to disk
                    save_path = os.path.join(
                        output_dir, f"FN_{fn_count:02d}_{filename}"
                    )
                    composite.save(save_path)

                    # ── Display inline in Kaggle / Jupyter ──────────────────
                    if IN_NOTEBOOK:
                        print(f"\n  FN #{fn_count:02d} | file: {filename} "
                              f"| pred_prob: {pred_prob:.4f}")
                        ipy_display(composite)
                    # ────────────────────────────────────────────────────────

                    results.append({
                        'filename':      filename,
                        'true_label':    true_label,
                        'predicted_prob': pred_prob,
                    })

                    fn_count += 1
                    pbar.update(1)

                    if fn_count >= max_samples:
                        break

                global_idx += 1

            if fn_count >= max_samples:
                break

    gradcam.remove_hooks()

    # Summary CSV
    csv_path = os.path.join(output_dir, 'fn_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['filename', 'true_label', 'predicted_prob']
        )
        writer.writeheader()
        writer.writerows(results)

    if fn_count == 0:
        print("\n[GradCAM] No false negatives found at the given threshold. "
              "Try lowering the threshold value.")
    else:
        print(f"\nGradCAM complete — {fn_count} FN cases saved to '{output_dir}/'")