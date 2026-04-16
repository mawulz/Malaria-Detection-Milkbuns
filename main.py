import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
import os
from tqdm import tqdm

from dataset import prepare_data
from model import build_model
from evaluate import calculate_metrics, profile_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_early_layers(model):
    """Freezes torchvision EfficientNet stem and first two MBConv blocks"""
    for idx in range(3):
        for param in model.features[idx].parameters():
            param.requires_grad = False

def unfreeze_all(model):
    """Unfreezes all layers for fine-tuning"""
    for param in model.parameters():
        param.requires_grad = True

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        # Mixed Precision
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item() * images.size(0)
        
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    metrics = calculate_metrics(all_labels, all_preds)
    return running_loss / len(loader.dataset), metrics

def train_model(seed, data_dir='./'):
    set_seed(seed)
    print(f"\n{'='*40}\nStarting Training with Seed: {seed}\n{'='*40}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = prepare_data(data_dir=data_dir, batch_size=32)
    
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    
    max_epochs = 120 # 20 stage 1 + 100 stage 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    patience = 10
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_model_state = None
    
    print("\n--- Stage 1: Freezing early layers (1-20) ---")
    freeze_early_layers(model)
    stage1_epochs = 20
    
    for epoch in range(1, stage1_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['F1-Score']:.4f}")
        with open('training_history.csv', 'a', encoding='utf-8') as f:
            f.write(f"{seed},{epoch},{train_loss:.4f},{val_loss:.4f},{val_metrics['F1-Score']:.4f},{val_metrics['Recall']:.4f}\n")
        
        if val_metrics['F1-Score'] > best_val_f1:
            best_val_f1 = val_metrics['F1-Score']
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
    print("\n--- Stage 2: Unfreezing all layers (Fine-tuning) ---")
    unfreeze_all(model)
    stage2_epochs = 100
    
    for epoch in range(stage1_epochs + 1, stage1_epochs + stage2_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['F1-Score']:.4f}")
        with open('training_history.csv', 'a', encoding='utf-8') as f:
            f.write(f"{seed},{epoch},{train_loss:.4f},{val_loss:.4f},{val_metrics['F1-Score']:.4f},{val_metrics['Recall']:.4f}\n")
        
        if val_metrics['F1-Score'] > best_val_f1:
            best_val_f1 = val_metrics['F1-Score']
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    print("\nEvaluating best model on test set...")
    model.load_state_dict(best_model_state)
    test_loss, test_metrics = eval_epoch(model, test_loader, criterion, device)
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return test_metrics

def main():
    seeds = [42, 123, 2026]
    results = []
    
    with open('training_history.csv', 'w', encoding='utf-8') as f:
        f.write("Seed,Epoch,Train_Loss,Val_Loss,Val_F1_Score,Val_Recall\n")
    
    print("Profiling Model before starting training pipeline...")
    temp_model = build_model()
    profile_model(temp_model)
    del temp_model
    torch.cuda.empty_cache()
    
    for seed in seeds:
        res = train_model(seed)
        results.append(res)
        
    print("\n\n" + "="*50)
    print("FINAL RESULTS ACROSS 3 SEEDS")
    print("="*50)
    
    keys = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC-ROC']
    with open('final_results.txt', 'w', encoding='utf-8') as f_out:
        f_out.write("FINAL RESULTS ACROSS 3 SEEDS\n")
        f_out.write("="*50 + "\n")
        for k in keys:
            vals = [r[k] for r in results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            res_str = f"{k}: {mean_val:.4f} ± {std_val:.4f}"
            print(res_str)
            f_out.write(res_str + "\n")

if __name__ == "__main__":
    main()
