from scipy import stats
import numpy as np

# Exact evaluation results across all 5 seeds (42, 123, 2026, 7, 17)
seeds = [42, 123, 2026, 7, 17]

metrics = {
    'Accuracy': {
        'baseline': [0.9681, 0.9702, 0.9644, 0.9681, 0.9677],
        'cbam':     [0.9670, 0.9663, 0.9655, 0.9626, 0.9663],
        'ecanet':   [0.9677, 0.9655, 0.9629, 0.9644, 0.9677]
    },
    'Recall': {
        'baseline': [0.9710, 0.9702, 0.9652, 0.9681, 0.9702],
        'cbam':     [0.9702, 0.9681, 0.9659, 0.9695, 0.9673],
        'ecanet':   [0.9702, 0.9695, 0.9673, 0.9659, 0.9659]
    },
    'Precision': {
        'baseline': [0.9654, 0.9702, 0.9638, 0.9681, 0.9653],
        'cbam':     [0.9640, 0.9646, 0.9652, 0.9563, 0.9652],
        'ecanet':   [0.9653, 0.9619, 0.9590, 0.9631, 0.9695]
    },
    'F1-Score': {
        'baseline': [0.9682, 0.9702, 0.9645, 0.9681, 0.9678],
        'cbam':     [0.9671, 0.9663, 0.9655, 0.9629, 0.9663],
        'ecanet':   [0.9678, 0.9657, 0.9631, 0.9645, 0.9677]
    },
    'AUC-ROC': {
        'baseline': [0.9941, 0.9937, 0.9940, 0.9940, 0.9941],
        'cbam':     [0.9940, 0.9937, 0.9934, 0.9935, 0.9934],
        'ecanet':   [0.9943, 0.9934, 0.9928, 0.9938, 0.9935]
    }
}

def print_test(name_exp, key_exp, key_ref='baseline'):
    print("=" * 80)
    print(f"PAIRED STATISTICAL TEST: {name_exp} vs {key_ref.upper()} (N = 5 Seeds)")
    print("=" * 80)
    print(f"{'Metric':<12} {'Ref Mean':>10} {'Exp Mean':>10} {'Mean Diff':>12} {'t-stat':>9} {'p-value':>9} {'Cohen d':>9} {'Sig(p<0.05)':>14}")
    print("-" * 80)
    for m, vals in metrics.items():
        ref = np.array(vals[key_ref])
        exp = np.array(vals[key_exp])
        diff = exp - ref
        t_stat, p_val = stats.ttest_rel(exp, ref)
        cohen_d = diff.mean() / (diff.std(ddof=1) if diff.std(ddof=1) != 0 else 1e-9)
        sig = "Significant" if p_val < 0.05 else "Not Sig (ns)"
        print(f"{m:<12} {ref.mean():>10.4f} {exp.mean():>10.4f} {diff.mean():>+12.4f} {t_stat:>9.4f} {p_val:>9.4f} {cohen_d:>9.4f} {sig:>14}")

print_test("HYBRID (ECA-Net)", "ecanet", "baseline")
print()
print_test("HYBRID (CBAM)", "cbam", "baseline")
print()
print_test("HYBRID (ECA-Net)", "ecanet", "cbam")