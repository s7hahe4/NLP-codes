import torch
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Load the Baseline Model and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./distilbert_baseline"
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# 2. Load the Dataset (Test set)
dataset = load_dataset("clinc_oos", "plus")
test_data = dataset["test"]

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt")

# 3. Monte-Carlo Dropout Function
def get_mcd_predictions(model, batch, n_iterations=10):
    model.train() # Enable dropout during inference
    batch = {k: v.to(device) for k, v in batch.items()}
    
    all_logits = []
    for _ in range(n_iterations):
        with torch.no_grad():
            outputs = model(**batch)
            all_logits.append(torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy())
    
    return np.array(all_logits) # Shape: [n_iterations, batch_size, num_labels]

# 4. Processing and Uncertainty Calculation
results = []
print("Starting Phase 2: Mapping Uncertainty with MCD...")

for i in tqdm(range(len(test_data))):
    inputs = tokenizer(test_data[i]["text"], padding="max_length", truncation=True, return_tensors="pt")
    probs = get_mcd_predictions(model, inputs)
    
    # Calculate Mean Probability across iterations
    mean_probs = np.mean(probs, axis=0) # [1, 151]
    
    # Calculate Predictive Entropy: -sum(p * log(p))
    # Higher entropy = Higher uncertainty
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    
    results.append({
        "text": test_data[i]["text"],
        "label": test_data[i]["intent"],
        "prediction": np.argmax(mean_probs),
        "entropy": entropy
    })

# 5. Save the most "Uncertain" samples for the Teacher LLM
# ... [Keep your Step 1-4 the same] ...

# 5. Fix and Save the most "Uncertain" samples
# We convert numpy types to standard python types using .item()
json_ready_results = []
for s in results:
    json_ready_results.append({
        "text": s["text"],
        "label": int(s["label"]), # Convert to standard int
        "prediction": int(s["prediction"]), # Convert np.int64 to standard int
        "entropy": float(s["entropy"]) # Convert np.float to standard float
    })

# Sort by highest entropy
uncertain_samples = sorted(json_ready_results, key=lambda x: x['entropy'], reverse=True)

print(f"\nTop 3 Most Uncertain Samples (Verified):")
for s in uncertain_samples[:3]:
    print(f"Text: {s['text']} | Entropy: {s['entropy']:.4f}")

# Save these to a file for Phase 3 (LLM Generation)
import json
with open("uncertain_samples.json", "w") as f:
    json.dump(uncertain_samples, f, indent=4)

print("Success! 'uncertain_samples.json' is now saved and ready for the LLM Teacher.")