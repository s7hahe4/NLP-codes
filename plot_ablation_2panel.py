import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

configs = [
    "Baseline",
    "Baseline +\n20x Loss",
    "Baseline +\n40 CF",
    "Full AD-COG\n(Proposed)",
    "Generic Aug.\n(N=40)"
]

# Means & Std
full_auroc_mean, full_auroc_std = [0.9739, 0.9748, 0.9769, 0.9766, 0.9752], [0.0011, 0.0008, 0.0007, 0.0012, 0.0022]
stress_auroc_mean, stress_auroc_std = [0.8180, 0.8183, 0.8341, 0.8312, 0.8145], [0.0075, 0.0085, 0.0051, 0.0070, 0.0063]

full_fpr_mean, full_fpr_std = [0.0910, 0.0852, 0.0831, 0.0801, 0.0887], [0.0048, 0.0053, 0.0019, 0.0035, 0.0042]
stress_fpr_mean, stress_fpr_std = [0.6762, 0.6681, 0.6651, 0.6621, 0.7042], [0.0576, 0.0460, 0.0358, 0.0664, 0.0337]

x = np.arange(len(configs))
width = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

# --- Panel 1: AUROC Comparison ---
rects1 = ax1.bar(x - width/2, full_auroc_mean, width, yerr=full_auroc_std, capsize=4, label='Full Test Set', color='#4c72b0', edgecolor='black', alpha=0.85)
rects2 = ax1.bar(x + width/2, stress_auroc_mean, width, yerr=stress_auroc_std, capsize=4, label='Stress Test Set (Top-500)', color='#55a868', edgecolor='black', alpha=0.85)
ax1.set_ylabel('AUROC Score (↑ Higher is Better)', fontweight='bold')
ax1.set_title('AUROC Performance Across Configurations', fontweight='bold', fontsize=12, pad=25)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=8.5)
ax1.set_ylim(0.70, 1.08)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# Place labels horizontal, non-overlapping, above error caps
for rect, mean, std in zip(rects1, full_auroc_mean, full_auroc_std):
    ax1.text(rect.get_x() + rect.get_width()/2., mean + std + 0.012, f'{mean:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2b4566')
for rect, mean, std in zip(rects2, stress_auroc_mean, stress_auroc_std):
    ax1.text(rect.get_x() + rect.get_width()/2., mean + std + 0.012, f'{mean:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2a5934')

# --- Panel 2: FPR95 Comparison ---
rects3 = ax2.bar(x - width/2, full_fpr_mean, width, yerr=full_fpr_std, capsize=4, label='Full Test Set', color='#c44e52', edgecolor='black', alpha=0.85)
rects4 = ax2.bar(x + width/2, stress_fpr_mean, width, yerr=stress_fpr_std, capsize=4, label='Stress Test Set (Top-500)', color='#8172b0', edgecolor='black', alpha=0.85)
ax2.set_ylabel('FPR95 Rate (↓ Lower is Better)', fontweight='bold')
ax2.set_title('FPR95 Performance Across Configurations', fontweight='bold', fontsize=12, pad=25)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=8.5)
ax2.set_ylim(0.0, 0.95)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.5)

for rect, mean, std in zip(rects3, full_fpr_mean, full_fpr_std):
    ax2.text(rect.get_x() + rect.get_width()/2., mean + std + 0.015, f'{mean:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#6e282a')
for rect, mean, std in zip(rects4, stress_fpr_mean, stress_fpr_std):
    ax2.text(rect.get_x() + rect.get_width()/2., mean + std + 0.015, f'{mean:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#453a63')

plt.tight_layout()
plt.savefig('d:/Research paper 1 nlp codes/ablation_matrix_2panel.png', dpi=300, bbox_inches='tight')
print("Successfully regenerated 2-panel plot: ablation_matrix_2panel.png")
