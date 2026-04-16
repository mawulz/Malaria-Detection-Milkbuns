import time
import torch
from thop import profile
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred_probs):
    y_true = np.array(y_true).flatten()
    y_pred_probs = np.array(y_pred_probs).flatten()
    
    # Binary predictions (threshold 0.5)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    # derived manually from confusion matrix if needed, but sklearn does it
    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
    
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5 # only happens if 1 class in y_true
        
    return {
        "Accuracy": acc,
        "Recall": recall,
        "Precision": prec,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP
    }

def profile_model(model, input_size=(1, 3, 224, 224), device='cpu'):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    model.to(device)
    
    # 1. Parameter count
    param_count = sum(p.numel() for p in model.parameters())
    
    # 2. Model size (MB)
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # 3. FLOPs using thop
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2  # FLOPs is roughly 2 * MACs
    
    # 4. Inference time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    
    # gpu warmup
    if device == 'cuda':
        for _ in range(10):
            _ = model(dummy_input)
            
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        avg_infer_time = np.mean(timings)
    else:
        # fallback to cpu timing
        for _ in range(5):
            _ = model(dummy_input)
        start_t = time.time()
        for _ in range(repetitions):
            _ = model(dummy_input)
        avg_infer_time = (time.time() - start_t) * 1000 / repetitions
        
    print(f"--- Model Profiling ---")
    print(f"Parameter Count: {param_count:,}")
    print(f"Model Size:      {size_mb:.2f} MB")
    print(f"FLOPs:           {flops:,}")
    print(f"Inference Time:  {avg_infer_time:.2f} ms")
    
    return param_count, size_mb, flops, avg_infer_time
