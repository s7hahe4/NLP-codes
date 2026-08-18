# full_test_auroc_compare.py
import numpy as np
import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def detect_oos_label_from_dataset(test_split, label_key="intent"):
    labels = np.array([int(x[label_key]) for x in test_split], dtype=np.int64)
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])  # most frequent label (OOS in CLINC plus)


@torch.no_grad()
def compute_pid_scores(model, tokenizer, texts, oos_label, device="cpu", batch_size=32, max_length=128):
    model.eval()
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)  # [B, num_labels]
        pid = 1.0 - probs[:, oos_label]            # P(ID) = 1 - P(OOS)
        scores.extend(pid.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)


def main():
    baseline_path = "./distilbert_baseline"
    adcog_path = "./ad_cog_final_model"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # IMPORTANT: Use the same dataset you used in Phase 1/2
    test_data = load_dataset("clinc_oos", "plus")["test"]
    label_key = "intent"

    # Auto-detect OOS label (will be 42 for your setup)
    oos_label = detect_oos_label_from_dataset(test_data, label_key=label_key)
    print(f"Auto-detected OOS label index = {oos_label}")

    texts = [x["text"] for x in test_data]
    y = np.array([0 if int(x[label_key]) == oos_label else 1 for x in test_data], dtype=np.int64)

    n_id = int(y.sum())
    n_oos = int((y == 0).sum())
    print(f"Full test composition => ID: {n_id}, OOS: {n_oos}")

    # tokenizer can be loaded from baseline (same vocab)
    tokenizer = DistilBertTokenizer.from_pretrained(baseline_path)

    baseline_model = DistilBertForSequenceClassification.from_pretrained(baseline_path).to(device)
    adcog_model = DistilBertForSequenceClassification.from_pretrained(adcog_path).to(device)

    base_scores = compute_pid_scores(baseline_model, tokenizer, texts, oos_label=oos_label, device=device)
    cog_scores = compute_pid_scores(adcog_model, tokenizer, texts, oos_label=oos_label, device=device)

    base_auroc = roc_auc_score(y, base_scores)
    cog_auroc = roc_auc_score(y, cog_scores)

    def get_fpr95(labels, scores):
        fpr, tpr, _ = roc_curve(labels, scores)
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            return fpr[idx[0]]
        return float("nan")

    base_fpr95 = get_fpr95(y, base_scores)
    cog_fpr95 = get_fpr95(y, cog_scores)

    print("\n--- Full-test AUROC & FPR95 (Corrected P(ID)) ---")
    print(f"Baseline AUROC: {base_auroc:.4f} | FPR95: {base_fpr95:.4f}")
    print(f"AD-COG  AUROC: {cog_auroc:.4f} | FPR95: {cog_fpr95:.4f}")
    print(f"Gain:          {cog_auroc - base_auroc:+.4f} | FPR95 Diff: {cog_fpr95 - base_fpr95:+.4f}")

    # Save a simple bar chart (optional but helpful)
    plt.figure()
    plt.bar(["Baseline", "AD-COG"], [base_auroc, cog_auroc])
    plt.ylim(0.0, 1.0)
    plt.title("Full-test AUROC (Corrected P(ID))")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.savefig("full_test_auroc_bar.png", dpi=200)
    plt.close()
    print("\nSaved: full_test_auroc_bar.png")


if __name__ == "__main__":
    main()