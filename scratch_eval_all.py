import torch
import torch.nn as nn
import numpy as np
import random
import json
from scipy import stats
from torch.utils.data import DataLoader
from dataset import MalariaDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from evaluate import calculate_metrics

import model_baseline
import model as model_cbam
import model_ecanet

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

seeds = [42, 123, 2026, 7, 17]

def eval_single_model(model_fn, ckpt_path, seed):
    set_seed(seed)
    model = model_fn().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds_val = []
    all_labels_val = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
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

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds, threshold=best_thresh)
    return metrics, best_thresh

all_results = {'baseline': {}, 'cbam': {}, 'ecanet': {}}

print("Evaluating Baseline Models...")
for seed in seeds:
    ckpt = f"checkpoint_baseline_seed_{seed}.pth"
    m, th = eval_single_model(model_baseline.build_model, ckpt, seed)
    all_results['baseline'][seed] = m
    print(f"Baseline Seed {seed}: Acc={m['Accuracy']:.4f}, Rec={m['Recall']:.4f}, Prec={m['Precision']:.4f}, F1={m['F1-Score']:.4f}, AUC={m['AUC-ROC']:.4f} (Thresh={th:.2f})")

print("\nEvaluating CBAM Models...")
for seed in seeds:
    ckpt = f"best_model_seed{seed}.pt"
    m, th = eval_single_model(model_cbam.build_model, ckpt, seed)
    all_results['cbam'][seed] = m
    print(f"CBAM Seed {seed}: Acc={m['Accuracy']:.4f}, Rec={m['Recall']:.4f}, Prec={m['Precision']:.4f}, F1={m['F1-Score']:.4f}, AUC={m['AUC-ROC']:.4f} (Thresh={th:.2f})")

print("\nEvaluating ECA-Net Models...")
for seed in seeds:
    ckpt = f"best_model_ecanet_seed{seed}.pt"
    m, th = eval_single_model(model_ecanet.build_model, ckpt, seed)
    all_results['ecanet'][seed] = m
    print(f"ECA-Net Seed {seed}: Acc={m['Accuracy']:.4f}, Rec={m['Recall']:.4f}, Prec={m['Precision']:.4f}, F1={m['F1-Score']:.4f}, AUC={m['AUC-ROC']:.4f} (Thresh={th:.2f})")

with open('scratch_all_eval_metrics.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nSaved metrics to scratch_all_eval_metrics.json")
