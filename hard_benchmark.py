import torch
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset

# 1. Setup
baseline_path = "./distilbert_baseline"
adcog_path = "./ad_cog_final_model"
tokenizer = DistilBertTokenizer.from_pretrained(baseline_path)

def get_id_prob(model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        return 1.0 - probs[150]

# 2. Build a Balanced Hard Benchmark
# We take the Hardest OOS samples and compare them to Easy ID samples
with open("uncertain_samples.json", "r") as f:
    hard_oos = [s for s in json.load(f) if s['label'] == 150][:50]

easy_id = load_dataset("clinc/clinc_oos", "plus")["test"].select(range(50))

models = {
    "Baseline": DistilBertForSequenceClassification.from_pretrained(baseline_path),
    "AD-COG": DistilBertForSequenceClassification.from_pretrained(adcog_path)
}

results = {"Baseline": [], "AD-COG": []}
ground_truth = ([0] * len(hard_oos)) + ([1] * len(easy_id))

# 3. Process
for name, model in models.items():
    model.eval()
    print(f"Profiling {name}...")
    # Process Hard OOS
    for s in hard_oos:
        results[name].append(get_id_prob(model, s['text']))
    # Process Easy ID
    for s in easy_id:
        results[name].append(get_id_prob(model, s['text']))

# 4. Generate Graph
plt.figure(figsize=(8, 6))
for name, color in zip(["Baseline", "AD-COG"], ["red", "green"]):
    auc = roc_auc_score(ground_truth, results[name])
    fpr, tpr, _ = roc_curve(ground_truth, results[name])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=color, linewidth=2)
    print(f"{name} Hard-AUROC: {auc:.4f}")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Hard-Sample Survival Analysis')
plt.legend()
plt.savefig("survival_graph_v2.png")
print("\nSuccess! 'survival_graph_v2.png' is ready for your paper.")