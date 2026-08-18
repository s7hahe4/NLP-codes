import matplotlib.pyplot as plt
import numpy as np

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Data from Table 1
configs = [
    "Baseline",
    "Baseline +\n20x Loss ONLY",
    "Baseline +\n40 CF ONLY",
    "Full AD-COG\n(Proposed)",
    "Generic Aug.\n(Matched N=40)"
]

# Means and Standard Deviations
full_auroc_mean = [0.9739, 0.9748, 0.9769, 0.9766, 0.9752]
full_auroc_std  = [0.0011, 0.0008, 0.0007, 0.0012, 0.0022]

full_fpr_mean   = [0.0910, 0.0852, 0.0831, 0.0801, 0.0887]
full_fpr_std    = [0.0048, 0.0053, 0.0019, 0.0035, 0.0042]

stress_auroc_mean = [0.8180, 0.8183, 0.8341, 0.8312, 0.8145]
stress_auroc_std  = [0.0075, 0.0085, 0.0051, 0.0070, 0.0063]

stress_fpr_mean   = [0.6762, 0.6681, 0.6651, 0.6621, 0.7042]
stress_fpr_std    = [0.0576, 0.0460, 0.0358, 0.0664, 0.0337]

# Distinct palette
colors = ['#7f7f7f', '#a6c8e0', '#4682b4', '#1f77b4', '#d62728']

# Create 2x2 Subplots Figure with ample vertical height
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
fig.suptitle('8-Seed Evaluation Matrix Across 5 Configurations (Mean ± Std)', fontsize=15, fontweight='bold', y=0.98)

x = np.arange(len(configs))
bar_width = 0.55

# --- Subplot 1: Full Test AUROC (Higher is Better) ---
ax1 = axes[0, 0]
bars1 = ax1.bar(x, full_auroc_mean, yerr=full_auroc_std, capsize=5, color=colors, width=bar_width, edgecolor='black', alpha=0.9)
ax1.set_title('(a) Full Test AUROC (↑ Higher is Better)', fontweight='bold', pad=12)
ax1.set_ylim(0.965, 0.984)
ax1.set_ylabel('AUROC Score')
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=8.5)
for bar, mean, std in zip(bars1, full_auroc_mean, full_auroc_std):
    # Position text cleanly ABOVE the top error bar cap
    y_pos = mean + std + 0.0008
    ax1.text(bar.get_x() + bar.get_width()/2, y_pos, f'{mean:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Subplot 2: Full Test FPR95 (Lower is Better) ---
ax2 = axes[0, 1]
bars2 = ax2.bar(x, full_fpr_mean, yerr=full_fpr_std, capsize=5, color=colors, width=bar_width, edgecolor='black', alpha=0.9)
ax2.set_title('(b) Full Test FPR95 (↓ Lower is Better)', fontweight='bold', pad=12)
ax2.set_ylim(0.065, 0.112)
ax2.set_ylabel('FPR95 Rate')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=8.5)
for bar, mean, std in zip(bars2, full_fpr_mean, full_fpr_std):
    y_pos = mean + std + 0.0015
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{mean:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Subplot 3: Failure Stress Test AUROC (Higher is Better) ---
ax3 = axes[1, 0]
bars3 = ax3.bar(x, stress_auroc_mean, yerr=stress_auroc_std, capsize=5, color=colors, width=bar_width, edgecolor='black', alpha=0.9)
ax3.set_title('(c) Failure Stress Test AUROC (↑ Higher is Better)', fontweight='bold', pad=12)
ax3.set_ylim(0.795, 0.860)
ax3.set_ylabel('AUROC Score')
ax3.set_xticks(x)
ax3.set_xticklabels(configs, fontsize=8.5)
for bar, mean, std in zip(bars3, stress_auroc_mean, stress_auroc_std):
    y_pos = mean + std + 0.002
    ax3.text(bar.get_x() + bar.get_width()/2, y_pos, f'{mean:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Subplot 4: Failure Stress Test FPR95 (Lower is Better) ---
ax4 = axes[1, 1]
bars4 = ax4.bar(x, stress_fpr_mean, yerr=stress_fpr_std, capsize=5, color=colors, width=bar_width, edgecolor='black', alpha=0.9)
ax4.set_title('(d) Failure Stress Test FPR95 (↓ Lower is Better)', fontweight='bold', pad=12)
# Generous y-limit so high error caps + text fit without hitting the top boundary
ax4.set_ylim(0.550, 0.840)
ax4.set_ylabel('FPR95 Rate')
ax4.set_xticks(x)
ax4.set_xticklabels(configs, fontsize=8.5)
for bar, mean, std in zip(bars4, stress_fpr_mean, stress_fpr_std):
    y_pos = mean + std + 0.015
    ax4.text(bar.get_x() + bar.get_width()/2, y_pos, f'{mean:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('d:/Research paper 1 nlp codes/ablation_matrix_plot.png', dpi=300, bbox_inches='tight')
print("Successfully regenerated 2x2 plot: ablation_matrix_plot.png")
