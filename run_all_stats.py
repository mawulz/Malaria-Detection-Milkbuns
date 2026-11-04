import numpy as np
import scipy.stats as stats

seeds = [42, 123, 2026, 7, 17]

metrics_data = {
    'Accuracy': {
        'Baseline': [0.9681, 0.9702, 0.9644, 0.9681, 0.9677],
        'CBAM':     [0.9670, 0.9663, 0.9655, 0.9626, 0.9663],
        'ECANet':   [0.9677, 0.9655, 0.9629, 0.9644, 0.9677],
    },
    'Recall': {
        'Baseline': [0.9710, 0.9702, 0.9652, 0.9681, 0.9702],
        'CBAM':     [0.9702, 0.9681, 0.9659, 0.9695, 0.9673],
        'ECANet':   [0.9702, 0.9695, 0.9673, 0.9659, 0.9659],
    },
    'Precision': {
        'Baseline': [0.9654, 0.9702, 0.9638, 0.9681, 0.9653],
        'CBAM':     [0.9640, 0.9646, 0.9652, 0.9563, 0.9652],
        'ECANet':   [0.9653, 0.9619, 0.9590, 0.9631, 0.9695],
    },
    'F1-Score': {
        'Baseline': [0.9682, 0.9702, 0.9645, 0.9681, 0.9678],
        'CBAM':     [0.9671, 0.9663, 0.9655, 0.9629, 0.9663],
        'ECANet':   [0.9678, 0.9657, 0.9631, 0.9645, 0.9677],
    },
    'AUC-ROC': {
        'Baseline': [0.9941, 0.9937, 0.9940, 0.9940, 0.9941],
        'CBAM':     [0.9940, 0.9937, 0.9934, 0.9935, 0.9934],
        'ECANet':   [0.9943, 0.9934, 0.9928, 0.9938, 0.9935],
    }
}

def analyze_pair(name_a, name_b, a_vals, b_vals):
    a = np.array(a_vals)
    b = np.array(b_vals)
    diff = b - a # b compared to a (e.g. ECANet - Baseline)
    t_stat, p_val = stats.ttest_rel(b, a) # paired t-test
    
    # Wilcoxon signed rank test
    try:
        w_stat, w_pval = stats.wilcoxon(b, a)
    except Exception as e:
        w_stat, w_pval = np.nan, np.nan
        
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohen_d = mean_diff / std_diff if std_diff != 0 else 0.0
    # Hedges' g correction for small samples n=5
    n = len(diff)
    hedges_g = cohen_d * (1 - 3 / (4 * (n - 1) - 1))
    
    return {
        'mean_a': np.mean(a),
        'std_a': np.std(a, ddof=1),
        'mean_b': np.mean(b),
        'std_b': np.std(b, ddof=1),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_stat': t_stat,
        'p_val': p_val,
        'w_pval': w_pval,
        'cohen_d': cohen_d,
        'hedges_g': hedges_g,
        'sig_05': p_val < 0.05
    }

print("="*80)
print("1. STATISTICAL TEST: HYBRID ECA-NET vs BASELINE (5 SEEDS)")
print("="*80)
print(f"{'Metric':<12} {'Baseline Mean±Std':<20} {'ECA-Net Mean±Std':<20} {'Diff (ECA-Base)':<16} {'t-stat':<10} {'p-value':<10} {'Cohen d':<10} {'Sig(p<0.05)':<10}")
print("-" * 110)

for metric, data in metrics_data.items():
    res = analyze_pair('Baseline', 'ECANet', data['Baseline'], data['ECANet'])
    sig_str = "YES (p<0.05)" if res['sig_05'] else "No (ns)"
    print(f"{metric:<12} {res['mean_a']:.4f}±{res['std_a']:.4f}       {res['mean_b']:.4f}±{res['std_b']:.4f}       {res['mean_diff']:>+10.4f}       {res['t_stat']:>8.4f}   {res['p_val']:>8.4f}   {res['cohen_d']:>8.4f}   {sig_str}")

print("\n" + "="*80)
print("2. STATISTICAL TEST: HYBRID CBAM vs BASELINE (5 SEEDS)")
print("="*80)
print(f"{'Metric':<12} {'Baseline Mean±Std':<20} {'CBAM Mean±Std':<20} {'Diff (CBAM-Base)':<16} {'t-stat':<10} {'p-value':<10} {'Cohen d':<10} {'Sig(p<0.05)':<10}")
print("-" * 110)

for metric, data in metrics_data.items():
    res = analyze_pair('Baseline', 'CBAM', data['Baseline'], data['CBAM'])
    sig_str = "YES (p<0.05)" if res['sig_05'] else "No (ns)"
    print(f"{metric:<12} {res['mean_a']:.4f}±{res['std_a']:.4f}       {res['mean_b']:.4f}±{res['std_b']:.4f}       {res['mean_diff']:>+10.4f}       {res['t_stat']:>8.4f}   {res['p_val']:>8.4f}   {res['cohen_d']:>8.4f}   {sig_str}")

print("\n" + "="*80)
print("3. DIRECT COMPARISON: HYBRID ECA-NET vs HYBRID CBAM (5 SEEDS)")
print("="*80)
print(f"{'Metric':<12} {'CBAM Mean±Std':<20} {'ECA-Net Mean±Std':<20} {'Diff (ECA-CBAM)':<16} {'t-stat':<10} {'p-value':<10} {'Cohen d':<10} {'Sig(p<0.05)':<10}")
print("-" * 110)

for metric, data in metrics_data.items():
    res = analyze_pair('CBAM', 'ECANet', data['CBAM'], data['ECANet'])
    sig_str = "YES (p<0.05)" if res['sig_05'] else "No (ns)"
    print(f"{metric:<12} {res['mean_a']:.4f}±{res['std_a']:.4f}       {res['mean_b']:.4f}±{res['std_b']:.4f}       {res['mean_diff']:>+10.4f}       {res['t_stat']:>8.4f}   {res['p_val']:>8.4f}   {res['cohen_d']:>8.4f}   {sig_str}")
