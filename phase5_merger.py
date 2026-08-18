# phase5_merger.py
import json
from datasets import load_dataset, Dataset, concatenate_datasets

# IMPORTANT: Use SAME dataset id as Phase 1/2
raw_dataset = load_dataset("clinc_oos", "plus")
train_original = raw_dataset["train"]

OOS_LABEL = 42  # <-- FIX: match your Phase-2 detected OOS label

with open("filtered_hard_data.json", "r", encoding="utf-8") as f:
    hard_data = json.load(f)

new_samples_dict = {
    "text": [item["text"] for item in hard_data],
    "intent": [OOS_LABEL for _ in hard_data],
}

# Ensure schema alignment (intent as ClassLabel)
train_hard = Dataset.from_dict(new_samples_dict, features=train_original.features)

train_augmented = concatenate_datasets([train_original, train_hard])

print(f"--- Dataset Merger Success ---")
print(f"Original Training Size: {len(train_original)}")
print(f"New 'Hard Counterfactuals' Added: {len(train_hard)}")
print(f"Final Augmented Training Size: {len(train_augmented)}")

train_augmented.save_to_disk("./augmented_train_dataset")
print("\nSuccess! Saved to './augmented_train_dataset'")