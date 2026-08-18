import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Style helper functions
def draw_box(ax, x, y, w, h, title, subtitle, bg_color='#eef4fb', border_color='#2b5c8f'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.15",
                                facecolor=bg_color, edgecolor=border_color, linewidth=1.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.68, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1a3048')
    ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=7.8, color='#334e68')

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#2b5c8f', edgecolor='#2b5c8f', width=1.5, headwidth=7, shrink=0.08),
                ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1a3048')

# Title
ax.text(7, 7.6, "AD-COG End-to-End Workflow Architecture", ha='center', va='center', fontsize=14, fontweight='bold', color='#0f2942')
ax.text(7, 7.3, "Active Failure Discovery, LLM Counterfactual Generation, and Weighted Retraining Pipeline", ha='center', va='center', fontsize=9, color='#486581')

# Row 1: Phases 1 to 3
draw_box(ax, 0.5, 4.8, 3.6, 1.8, "Phase 1: Baseline Expert", "DistilBERT Fine-Tuning\non CLINC150 (15,250 Train)", bg_color='#e3f2fd', border_color='#1565c0')
draw_arrow(ax, 4.1, 5.7, 5.0, 5.7)

draw_box(ax, 5.0, 4.8, 3.8, 1.8, "Phase 2: Active Discovery", "Monte Carlo Dropout (T=10)\nPredictive Entropy H(x) Mapping", bg_color='#fff3e0', border_color='#e65100')
draw_arrow(ax, 8.8, 5.7, 9.7, 5.7)

draw_box(ax, 9.7, 4.8, 3.8, 1.8, "Phase 3: LLM Generator", "Teacher LLM Prompts\nHard ID/OOS Logical Twins", bg_color='#f3e5f5', border_color='#6a1b9a')

# Vertical Down Arrow
draw_arrow(ax, 11.6, 4.8, 11.6, 3.6)

# Row 2: Phases 4 to 6
draw_box(ax, 9.7, 1.8, 3.8, 1.8, "Phase 4: Quality Filtering", "Profile against Baseline\nEntropy Filter: H(x) > 1.0", bg_color='#e8f5e9', border_color='#2e7d32')
draw_arrow(ax, 9.7, 2.7, 8.8, 2.7)

draw_box(ax, 5.0, 1.8, 3.8, 1.8, "Phase 5: Robust Retraining", "Merge N=40 Hard CFs\n20x OOS Weighted Loss Penalty", bg_color='#fffde7', border_color='#f57f17')
draw_arrow(ax, 5.0, 2.7, 4.1, 2.7)

draw_box(ax, 0.5, 1.8, 3.6, 1.8, "Phase 6: Multi-Seed Benchmark", "8 Seeds (N=8) GPU Evaluation\nAUROC, FPR95 & Wilcoxon Test", bg_color='#ede7f6', border_color='#4527a0')

# Legend / System Output Box at bottom
box_out = patches.FancyBboxPatch((3.2, 0.3), 7.6, 0.9, boxstyle="round,pad=0.1,rounding_size=0.1",
                                 facecolor='#f8f9fa', edgecolor='#9e9e9e', linestyle='--', linewidth=1.2)
ax.add_patch(box_out)
ax.text(7, 0.75, "Primary Outputs: Hardened Classifier | 12% FPR95 Reduction | Significant Gains (p = 0.0078)",
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1b5e20')

plt.tight_layout()
plt.savefig('d:/Research paper 1 nlp codes/ad_cog_workflow_diagram.png', dpi=300, bbox_inches='tight')
print("Successfully generated workflow architecture diagram: ad_cog_workflow_diagram.png")
