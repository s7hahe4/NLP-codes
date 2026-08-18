import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 1. Setup Paths
baseline_path = "./distilbert_baseline"
adcog_path = "./ad_cog_final_model"
tokenizer = DistilBertTokenizer.from_pretrained(baseline_path)

# 2. The Critical "Failure Points" (Ablation Test queries)
test_queries = [
    "i need to find a new babysitter", # Was STILL WRONG
    "report outage to my electric provider", # Was ✅ IMPROVED
    "fold my hand", # Was ✅ IMPROVED
    "what supplies do i need to clean my car", # Was STILL WRONG
    "i have sudden numbness in my arm with slurred speech" # New Medical Logic Trap
]

def get_prediction(model_path, text):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        pred_label = np.argmax(probs)
    return pred_label

print(f"{'Test Query':<50} | {'Baseline':<10} | {'AD-COG'}")
print("-" * 85)

for query in test_queries:
    base_label = get_prediction(baseline_path, query)
    cog_label = get_prediction(adcog_path, query)
    
    base_res = "OOS" if base_label == 150 else "ID"
    cog_res = "OOS (FIXED)" if cog_label == 150 else "ID"
    
    status = "🌟 PERFECT" if cog_label == 150 else "STILL WRONG"
    
    print(f"{query[:48]:<50} | {base_res:<10} | {cog_res:<10} | {status}")