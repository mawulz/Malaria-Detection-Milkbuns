import numpy as np
import scipy.stats as stats
import json

with open('scratch_all_eval_metrics.json', 'r') as f:
    data = json.load(f)

seeds = ['42', '123', '2026', '7', '17']
metrics = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC-ROC']

print(f"{'Seed':<8} | " + " | ".join([f"{m} (Base / CBAM / ECA)" for m in metrics]))
print("-" * 120)
for s in seeds:
    row = [f"Seed {s:<4}"]
    for m in metrics:
        b = data['baseline'][s][m]
        c = data['cbam'][s][m]
        e = data['ecanet'][s][m]
        row.append(f"{b:.4f} / {c:.4f} / {e:.4f}")
    print(" | ".join(row))

print("\n" + "="*80)
print("WILCOXON SIGNED-RANK TESTS (Two-Sided)")
print("="*80)
for m in metrics:
    b_vals = [data['baseline'][s][m] for s in seeds]
    c_vals = [data['cbam'][s][m] for s in seeds]
    e_vals = [data['ecanet'][s][m] for s in seeds]
    
    # ECA vs Base
    try:
        w_eb, p_eb = stats.wilcoxon(e_vals, b_vals)
    except Exception:
        w_eb, p_eb = np.nan, np.nan
        
    # CBAM vs Base
    try:
        w_cb, p_cb = stats.wilcoxon(c_vals, b_vals)
    except Exception:
        w_cb, p_cb = np.nan, np.nan

    # ECA vs CBAM
    try:
        w_ec, p_ec = stats.wilcoxon(e_vals, c_vals)
    except Exception:
        w_ec, p_ec = np.nan, np.nan

    print(f"{m:<12} | ECA vs Base: p={p_eb:.4f} | CBAM vs Base: p={p_cb:.4f} | ECA vs CBAM: p={p_ec:.4f}")
