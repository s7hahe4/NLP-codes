# failure_only_auroc_plot.py
import json
from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def read_uncertain_file(infile: str):
    data = json.loads(Path(infile).read_text(encoding="utf-8"))
    if not data:
        raise ValueError("Input JSON is empty.")
    # detect label key
    label_key = "label" if "label" in data[0] else "intent"
    return data, label_key


def detect_oos_label(data, label_key: str) -> int:
    """
    In CLINC150 test split:
    - each ID intent usually appears ~30 times
    - OOS appears much more (often 1000)
    So the most frequent label is a reliable OOS label in your Phase-2 file.
    """
    labels = [int(x[label_key]) for x in data]
    values, counts = np.unique(labels, return_counts=True)
    oos = int(values[np.argmax(counts)])
    return oos


def select_topk_mixed(data, label_key: str, oos_label: int, start_k=500, max_k=None):
    """
    Sort by entropy and pick top-k, but guarantee both classes exist (ID and OOS).
    If not, automatically increase k until both appear.
    """
    if max_k is None:
        max_k = len(data)

    data_sorted = sorted(data, key=lambda x: float(x.get("entropy", 0.0)), reverse=True)

    k = start_k
    while k <= max_k:
        subset = data_sorted[:k]
        y = np.array([0 if int(s[label_key]) == oos_label else 1 for s in subset], dtype=np.int64)
        n_id = int(y.sum())
        n_oos = int((y == 0).sum())
        if n_id > 0 and n_oos > 0:
            return subset, k, n_id, n_oos
        k = min(max_k, k + 50)  # grow gradually

    raise ValueError("Could not form a mixed subset containing both ID and OOS.")


@torch.no_grad()
def compute_pid_scores(model, tokenizer, texts, oos_label: int, device="cpu", batch_size=16, max_length=128):
    """
    Corrected scoring (same as your final_metrics_v2 idea):
    P(ID) = 1 - P(OOS)
    """
    scores = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)  # [B, num_labels]
        if oos_label >= probs.shape[1]:
            raise ValueError(f"oos_label={oos_label} is out of range for model outputs (num_labels={probs.shape[1]}).")
        pid = 1.0 - probs[:, oos_label]
        scores.extend(pid.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)


def plot_bar(baseline_auroc, adcog_auroc, outpath):
    plt.figure()
    plt.bar(["Baseline", "AD-COG"], [baseline_auroc, adcog_auroc])
    plt.ylim(0.0, 1.0)
    plt.title("Failure-only AUROC (Uncertainty/Failure Set)")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_roc(labels, base_scores, cog_scores, outpath):
    fpr_b, tpr_b, _ = roc_curve(labels, base_scores)
    fpr_c, tpr_c, _ = roc_curve(labels, cog_scores)

    auc_b = auc(fpr_b, tpr_b)
    auc_c = auc(fpr_c, tpr_c)

    plt.figure()
    plt.plot(fpr_b, tpr_b, label=f"Baseline (AUROC={auc_b:.4f})")
    plt.plot(fpr_c, tpr_c, label=f"AD-COG (AUROC={auc_c:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Failure-only Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    # ---- CONFIG (matches your folder screenshot) ----
    infile = "uncertain_samples.json"
    start_k = 500  # try top-50 first; script will increase if needed
    baseline_path = "./distilbert_baseline"
    adcog_path = "./ad_cog_final_model"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load Phase-2 uncertainty file
    data, label_key = read_uncertain_file(infile)

    # 2) Auto-detect which label index corresponds to OOS in YOUR saved file
    oos_label = detect_oos_label(data, label_key)
    print(f"Auto-detected OOS label index = {oos_label}")

    # 3) Select a mixed hard subset (must contain both ID and OOS)
    subset, used_k, n_id, n_oos = select_topk_mixed(data, label_key, oos_label, start_k=start_k)
    print(f"Subset selected: top-{used_k} by entropy => ID: {n_id}, OOS: {n_oos}")

    # Save the subset so you can show your supervisor
    out_subset = f"failure_set_top{used_k}.json"
    cleaned = [{
        "text": s["text"],
        "label": int(s[label_key]),
        "entropy": float(s.get("entropy", 0.0)),
        "prediction": int(s.get("prediction", -1)) if "prediction" in s else None,
    } for s in subset]
    Path(out_subset).write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    print(f"Saved failure subset to: {out_subset}")

    texts = [s["text"] for s in cleaned]
    labels = np.array([0 if s["label"] == oos_label else 1 for s in cleaned], dtype=np.int64)

    # 4) Load tokenizer + models
    tokenizer = DistilBertTokenizer.from_pretrained(baseline_path)
    baseline_model = DistilBertForSequenceClassification.from_pretrained(baseline_path).to(device)
    adcog_model = DistilBertForSequenceClassification.from_pretrained(adcog_path).to(device)

    # 5) Compute corrected P(ID) scores
    base_scores = compute_pid_scores(baseline_model, tokenizer, texts, oos_label=oos_label, device=device)
    cog_scores = compute_pid_scores(adcog_model, tokenizer, texts, oos_label=oos_label, device=device)

    # 6) AUROC and FPR95
    baseline_auroc = roc_auc_score(labels, base_scores)
    adcog_auroc = roc_auc_score(labels, cog_scores)

    def get_fpr95(lbls, scores):
        fpr, tpr, _ = roc_curve(lbls, scores)
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            return fpr[idx[0]]
        return float("nan")

    baseline_fpr95 = get_fpr95(labels, base_scores)
    adcog_fpr95 = get_fpr95(labels, cog_scores)

    print("\n--- Failure-only AUROC & FPR95 (Hard/Failure Set) ---")
    print(f"Baseline AUROC: {baseline_auroc:.4f} | FPR95: {baseline_fpr95:.4f}")
    print(f"AD-COG  AUROC: {adcog_auroc:.4f} | FPR95: {adcog_fpr95:.4f}")

    # 7) Plots
    plot_bar(baseline_auroc, adcog_auroc, "failure_only_auroc_bar.png")
    plot_roc(labels, base_scores, cog_scores, "failure_only_roc_curves.png")

    print("\nSaved plots:")
    print(" - failure_only_auroc_bar.png")
    print(" - failure_only_roc_curves.png")


if __name__ == "__main__":
    main()