import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. Load the AD-COG Robust Model
model_path = "./ad_cog_final_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model.eval()

# 2. Load CLINC150 Test Set
test_data = load_dataset("clinc/clinc_oos", "plus")["test"]

def get_id_probability(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        # AD-COG Logic: The probability of being In-Distribution is the sum of all probs EXCEPT label 150
        # or simply: 1.0 - probs[150]
        return 1.0 - probs[150]

print("Calculating Corrected AUROC for AD-COG (P(ID) Logic)...")
scores = []
labels = [] # 1 for In-Distribution, 0 for Out-of-Scope

for item in test_data:
    p_id = get_id_probability(item['text'])
    scores.append(p_id)
    labels.append(0 if item['intent'] == 150 else 1)

final_auroc = roc_auc_score(labels, scores)

print(f"\n--- REVISED RESEARCH RESULTS ---")
print(f"AD-COG AUROC (Corrected): {final_auroc:.4f}")
print(f"Interpretation: Your model's ability to separate logic-traps from real intents.")