import torch
import json
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

# 1. Load your Baseline Model from Phase 1
model_path = "./distilbert_baseline"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# 2. Load the LLM-generated data
with open("generated_data.json", "r") as f:
    new_samples = json.load(f)

print(f"Profiling {len(new_samples)} new counterfactuals for Phase 4 Quality Control...")

# 3. Filtering Logic
hard_samples = []
for item in tqdm(new_samples):
    inputs = tokenizer(item['text'], return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        # Calculate Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # RESEARCH RULE: Only keep samples that create High Uncertainty (Entropy > 1.0)
    if entropy > 1.0:
        item['baseline_entropy'] = float(entropy)
        hard_samples.append(item)

# 4. Save the "Hardened" dataset
with open("filtered_hard_data.json", "w") as f:
    json.dump(hard_samples, f, indent=4)

print(f"\n--- Phase 4 Results ---")
print(f"Total Evaluated: {len(new_samples)}")
print(f"Kept as 'Hard Counterfactuals': {len(hard_samples)}")
print(f"Filtered Out (Too Easy): {len(new_samples) - len(hard_samples)}")