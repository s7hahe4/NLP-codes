import torch
import numpy as np
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 1. Paths to the two models
baseline_path = "./distilbert_baseline"
adcog_path = "./ad_cog_final_model"

tokenizer = DistilBertTokenizer.from_pretrained(baseline_path)

# 2. Test Sentences (The ones that caused the highest uncertainty in Phase 2)
test_queries = [
    "i need to find a new babysitter",
    "report outage to my electric provider",
    "fold my hand",
    "what supplies do i need to clean my car",
    "i have a rash, what can i use for it"
]

def get_prediction(model_path, text):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        conf = np.max(probs)
        pred_label = np.argmax(probs)
    return pred_label, conf

print(f"{'Test Query':<40} | {'Baseline Pred':<15} | {'AD-COG Pred':<15} | {'Robustness Gain'}")
print("-" * 100)

for query in test_queries:
    base_label, base_conf = get_prediction(baseline_path, query)
    cog_label, cog_conf = get_prediction(adcog_path, query)
    
    # We want Cog Label to be 150 (Out-of-Scope)
    base_res = "OOS" if base_label == 150 else "Wrong/ID"
    cog_res = "CORRECT (OOS)" if cog_label == 150 else "STILL WRONG"
    
    status = "✅ IMPROVED" if (base_label != 150 and cog_label == 150) else "---"
    
    print(f"{query[:38]:<40} | {base_res:<15} | {cog_res:<15} | {status}")

print("\nFinal Step: Check the entropy levels. If AD-COG has lower entropy on these, your Teacher-Student loop is validated.")