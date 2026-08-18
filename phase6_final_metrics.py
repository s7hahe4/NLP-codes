import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. Load Model
model = DistilBertForSequenceClassification.from_pretrained("./ad_cog_final_model")
tokenizer = DistilBertTokenizer.from_pretrained("./ad_cog_final_model")
model.eval()

# 2. Load CLINC150 Test Set (includes OOS samples)
test_data = load_dataset("clinc/clinc_oos", "plus")["test"]

def get_max_conf(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return torch.max(probs).item()

# 3. Calculate AUROC
print("Calculating Full-Scale AUROC for AD-COG...")
confs = []
labels = [] # 1 for In-Distribution, 0 for Out-of-Scope

for item in test_data:
    conf = get_max_conf(item['text'])
    confs.append(conf)
    # Label 150 is OOS in CLINC150
    labels.append(0 if item['intent'] == 150 else 1)

auroc = roc_auc_score(labels, confs)
print(f"\n--- FINAL RESEARCH RESULTS ---")
print(f"AD-COG AUROC: {auroc:.4f}")
print("Note: SOTA for DistilBERT on CLINC150 is usually ~0.92. Compare yours!")