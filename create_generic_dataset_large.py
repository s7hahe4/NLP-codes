import torch
import json
import numpy as np
import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets

model_path = "./distilbert_baseline"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

with open("generic_generated_data_large.json", "r") as f:
    new_samples = json.load(f)

print(f"Profiling {len(new_samples)} generic counterfactuals for Quality Control...")

OOS_LABEL = 42

hard_samples = []
for item in tqdm(new_samples):
    inputs = tokenizer(item['text'], return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Entropy threshold
    if entropy > 1.0:
        item['baseline_entropy'] = float(entropy)
        # Fix label if it is -1 (OOS from LLM)
        if item['label'] == -1 or "DOMAIN_GAP" in item['logic']:
            item['label'] = OOS_LABEL
        hard_samples.append(item)

print(f"\nTotal Evaluated: {len(new_samples)}")
print(f"Kept as 'Hard Counterfactuals': {len(hard_samples)}")
print(f"Filtered Out (Too Easy): {len(new_samples) - len(hard_samples)}")

# STRICT MATCH N=40
if len(hard_samples) > 40:
    print(f"Subsampling exactly 40 hard samples from {len(hard_samples)} to match AD-COG dataset...")
    random.seed(42)
    hard_samples = random.sample(hard_samples, 40)
elif len(hard_samples) < 40:
    print(f"WARNING: Only {len(hard_samples)} passed the filter, which is less than 40. The matched N constraint will fail!")

raw_dataset = load_dataset("clinc_oos", "plus")
train_original = raw_dataset["train"]

new_samples_dict = {
    "text": [item["text"] for item in hard_samples],
    "intent": [item["label"] for item in hard_samples],
}

train_hard = Dataset.from_dict(new_samples_dict, features=train_original.features)
train_augmented = concatenate_datasets([train_original, train_hard])

print(f"Original Training Size: {len(train_original)}")
print(f"Final Augmented Training Size: {len(train_augmented)}")

# Overwrite the old generic dataset
train_augmented.save_to_disk("./generic_dataset")
print("\nSuccess! Saved exactly 40 samples to './generic_dataset'")
