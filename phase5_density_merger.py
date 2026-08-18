# phase5_density_merger.py
from datasets import load_from_disk, Dataset, concatenate_datasets

base_augmented = load_from_disk("./augmented_train_dataset")

OOS_LABEL = 42  # <-- FIX

new_batch = [
    # (keep your exact 30 samples here unchanged)
]

new_data_dict = {
    "text": new_batch,
    "intent": [OOS_LABEL for _ in new_batch],
}

density_addition = Dataset.from_dict(new_data_dict, features=base_augmented.features)
high_density_dataset = concatenate_datasets([base_augmented, density_addition])

print(f"--- Ablation Study: Density Increase ---")
print(f"Previous Augmented Size: {len(base_augmented)}")
print(f"New Density Samples Added: {len(density_addition)}")
print(f"Total High-Density Size: {len(high_density_dataset)}")

high_density_dataset.save_to_disk("./high_density_dataset")
print("Success! Dataset saved to './high_density_dataset'")