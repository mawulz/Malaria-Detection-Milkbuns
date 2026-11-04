import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

from dataset import prepare_data
from model_baseline import build_model
from evaluate import calculate_metrics, profile_model
from gradcam import visualise_false_negatives

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5, return_raw=False):
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
        
    metrics = calculate_metrics(all_labels, all_preds, threshold=threshold)
    if return_raw:
        return running_loss / len(loader.dataset), metrics, all_labels, all_preds
    return running_loss / len(loader.dataset), metrics

def train_model(seed, data_dir='./'):
    set_seed(seed)
    print(f"\n{'='*40}\nStarting Training with Seed: {seed}\n{'='*40}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = prepare_data(data_dir=data_dir, batch_size=32)
    
    model = build_model().to(device)
    pos_weight = torch.tensor([1.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    backbone_params = [p for n, p in model.named_parameters() if 'features' in n]
    head_params     = [p for n, p in model.named_parameters() if 'features' not in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 1e-4}
    ], weight_decay=1e-2)

    scaler = torch.amp.GradScaler('cuda')
    
    max_epochs = 80
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5)
    
    patience = 10
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_model_state = None
    
    print("\n--- Training: End-to-End Fine-Tuning (80 Epochs) ---")
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics['F1-Score'])
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['F1-Score']:.4f}")
        with open('training_history_baseline.csv', 'a', encoding='utf-8') as f:
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
    torch.save(best_model_state, f'checkpoint_baseline_seed_{seed}.pth')
    print(f"Saved baseline checkpoint: checkpoint_baseline_seed_{seed}.pth")

    # Sweep thresholds on validation set
    _, _, val_labels, val_preds = eval_epoch(model, val_loader, criterion, device, return_raw=True)
    best_thresh = 0.5
    best_recall = -1
    for t in np.arange(0.3, 0.65, 0.05):
        m = calculate_metrics(val_labels, val_preds, threshold=t)
        if m['Precision'] > 0.95 and m['Recall'] > best_recall:
            best_recall = m['Recall']
            best_thresh = t

    print(f"Selected Threshold: {best_thresh:.2f}")
    test_loss, test_metrics = eval_epoch(model, test_loader, criterion, device, threshold=best_thresh)
    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    visualise_false_negatives(model, test_loader, device, output_dir=f'gradcam_fn_baseline_seed{seed}', threshold=best_thresh)
    
    return test_metrics

def main():
    seeds = [42, 123, 2026, 7, 17, 31, 55, 89, 101, 256]
    results = []
    
    with open('training_history_baseline.csv', 'w', encoding='utf-8') as f:
        f.write("Seed,Epoch,Train_Loss,Val_Loss,Val_F1_Score,Val_Recall\n")
    
    print("Profiling Model before starting training pipeline...")
    temp_model = build_model()
    profile_model(temp_model, device='cuda' if torch.cuda.is_available() else 'cpu')
    del temp_model
    torch.cuda.empty_cache()
    
    for seed in seeds:
        res = train_model(seed)
        results.append(res)
        
    print("\n\n" + "="*50)
    print("FINAL RESULTS ACROSS 10 SEEDS")
    print("="*50)
    
    keys = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC-ROC']
    with open('final_results_baseline.txt', 'w', encoding='utf-8') as f_out:
        f_out.write("FINAL RESULTS ACROSS 10 SEEDS\n")
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
