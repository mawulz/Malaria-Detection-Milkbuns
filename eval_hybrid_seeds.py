if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import numpy as np
    import random
    from torch.utils.data import DataLoader
    from dataset import MalariaDataset
    from model import build_model
    from evaluate import calculate_metrics
    from torchvision import transforms
    from sklearn.model_selection import train_test_split

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = './'
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = MalariaDataset(root_dir=data_dir, transform=None)
    total_imgs = len(full_dataset)
    print(f"Total Images: {total_imgs}")

    targets = full_dataset.labels.tolist()
    train_idx, temp_idx = train_test_split(list(range(total_imgs)), test_size=0.2, stratify=targets, random_state=42)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_targets, random_state=42)

    val_dataset = MalariaDataset(root_dir=data_dir, transform=val_test_transform)
    val_dataset.image_paths = [full_dataset.image_paths[i] for i in val_idx]
    val_dataset.labels = full_dataset.labels[val_idx]

    test_dataset = MalariaDataset(root_dir=data_dir, transform=val_test_transform)
    test_dataset.image_paths = [full_dataset.image_paths[i] for i in test_idx]
    test_dataset.labels = full_dataset.labels[test_idx]

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    for seed in [42, 123, 2026, 7, 17]:
        set_seed(seed)
        sep = "=" * 50
        print(f"\n{sep}")
        print(f"HYBRID (EfficientNet+CBAM) - Seed {seed}")
        print(sep)

        model = build_model().to(device)
        ckpt_path = f"best_model_seed{seed}.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Sweep thresholds on val set
        all_preds_val = []
        all_labels_val = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds_val.extend(probs)
                all_labels_val.extend(labels.cpu().numpy())

        best_thresh = 0.5
        best_recall = -1
        for t in np.arange(0.3, 0.65, 0.05):
            m = calculate_metrics(all_labels_val, all_preds_val, threshold=t)
            if m['Precision'] > 0.95 and m['Recall'] > best_recall:
                best_recall = m['Recall']
                best_thresh = t

        print(f"Selected Threshold: {best_thresh:.2f}")

        # Evaluate on test set
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(all_labels, all_preds, threshold=best_thresh, print_report=True)
        print(f"\nTest Metrics (Seed {seed}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
