import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon

def get_fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return float("nan")

@torch.no_grad()
def compute_pid_scores(model, tokenizer, texts, oos_label: int, device="cpu", batch_size=32, max_length=128):
    scores = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)
        pid = 1.0 - probs[:, oos_label]
        scores.extend(pid.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)

def evaluate_model(model_path, texts, labels, oos_label, device):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    scores = compute_pid_scores(model, tokenizer, texts, oos_label=oos_label, device=device)
    auroc = roc_auc_score(labels, scores)
    fpr95 = get_fpr95(labels, scores)
    return auroc, fpr95

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Full Test Data
    test_data = load_dataset("clinc_oos", "plus")["test"]
    full_texts = [x["text"] for x in test_data]
    
    OOS_LABEL = 42
    full_labels = np.array([0 if int(x["intent"]) == OOS_LABEL else 1 for x in test_data], dtype=np.int64)

    # Load Failure Only Data (Top 500)
    failure_file = "failure_set_top500.json"
    if not Path(failure_file).exists():
        print(f"File {failure_file} not found. Run failure_only_auroc_plot.py first to generate it.")
        return
    
    with open(failure_file, "r") as f:
        failure_data = json.load(f)
    failure_texts = [s["text"] for s in failure_data]
    failure_labels = np.array([0 if int(s["label"]) == OOS_LABEL else 1 for s in failure_data], dtype=np.int64)

    seeds = [42, 123, 999, 7, 2024]
    configs = [
        ("Baseline", "none", 1.0),
        ("Baseline + 20x ONLY", "none", 20.0),
        ("Baseline + 40 CF ONLY", "counterfactual", 1.0),
        ("Full AD-COG", "counterfactual", 20.0),
        ("Generic Augmentation", "generic", 20.0)
    ]

    results = {c[0]: {"full_auroc": [], "full_fpr95": [], "fail_auroc": [], "fail_fpr95": []} for c in configs}

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        for name, data_arg, loss_arg in configs:
            model_dir = f"./models/model_data_{data_arg}_loss_{loss_arg}_seed_{seed}"
            if not Path(model_dir).exists():
                print(f"Warning: Model directory {model_dir} not found. Skipping...")
                continue
            
            full_auc, full_fpr = evaluate_model(model_dir, full_texts, full_labels, OOS_LABEL, device)
            fail_auc, fail_fpr = evaluate_model(model_dir, failure_texts, failure_labels, OOS_LABEL, device)
            
            results[name]["full_auroc"].append(full_auc)
            results[name]["full_fpr95"].append(full_fpr)
            results[name]["fail_auroc"].append(fail_auc)
            results[name]["fail_fpr95"].append(fail_fpr)
            
            print(f"[{name}] Full AUROC: {full_auc:.4f}, Fail AUROC: {fail_auc:.4f}")

    print("\n--- Significance Testing Results ---")
    ad_cog_full_auroc = np.array(results["Full AD-COG"]["full_auroc"])
    ad_cog_fail_auroc = np.array(results["Full AD-COG"]["fail_auroc"])

    for name in results.keys():
        print(f"\nConfiguration: {name}")
        
        full_auc_arr = np.array(results[name]["full_auroc"])
        full_fpr_arr = np.array(results[name]["full_fpr95"])
        fail_auc_arr = np.array(results[name]["fail_auroc"])
        fail_fpr_arr = np.array(results[name]["fail_fpr95"])
        
        print(f"  Full Test AUROC: {np.mean(full_auc_arr):.4f} +/- {np.std(full_auc_arr):.4f}")
        print(f"  Full Test FPR95: {np.mean(full_fpr_arr):.4f} +/- {np.std(full_fpr_arr):.4f}")
        print(f"  Fail Test AUROC: {np.mean(fail_auc_arr):.4f} +/- {np.std(fail_auc_arr):.4f}")
        print(f"  Fail Test FPR95: {np.mean(fail_fpr_arr):.4f} +/- {np.std(fail_fpr_arr):.4f}")
        
        if name != "Full AD-COG" and len(full_auc_arr) == 5:
            # Wilcoxon signed-rank test
            _, p_full = wilcoxon(ad_cog_full_auroc, full_auc_arr)
            _, p_fail = wilcoxon(ad_cog_fail_auroc, fail_auc_arr)
            print(f"  -> p-value vs Full AD-COG (Full Test): {p_full:.4f}")
            print(f"  -> p-value vs Full AD-COG (Fail Test): {p_fail:.4f}")

if __name__ == "__main__":
    main()
