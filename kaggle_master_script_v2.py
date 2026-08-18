import os
import torch
import torch.nn as nn
import random
import numpy as np
import json
import glob
import shutil
from pathlib import Path
from datasets import load_from_disk, load_dataset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon

# ============================================================
# CONFIGURATION — change seeds here only
# ============================================================
SEEDS = [42, 123, 999, 7, 2024, 314, 1337, 77]   # 8 seeds
CONFIGS = [
    ("Baseline",                "none",            1.0),
    ("Baseline + 20x ONLY",     "none",            20.0),
    ("Baseline + 40 CF ONLY",   "counterfactual",  1.0),
    ("Full AD-COG",             "counterfactual",  20.0),
    ("Generic Augmentation",    "generic",         20.0),
]
OOS_LABEL = 42
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return float("nan")

def is_model_complete(model_dir):
    return (os.path.exists(os.path.join(model_dir, "model.safetensors")) or
            os.path.exists(os.path.join(model_dir, "pytorch_model.bin")))

@torch.no_grad()
def compute_pid_scores(model, tokenizer, texts, oos_label, device, batch_size=32, max_length=128):
    scores = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)
        pid = 1.0 - probs[:, oos_label]
        scores.extend(pid.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)

def evaluate_model(model_path, texts, labels, oos_label, device):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    scores = compute_pid_scores(model, tokenizer, texts, oos_label, device)
    auroc = roc_auc_score(labels, scores)
    fpr95 = get_fpr95(labels, scores)
    return auroc, fpr95

def wilcoxon_p(a, b):
    """Safe Wilcoxon: returns p-value, or 1.0 if arrays are identical."""
    a, b = np.array(a), np.array(b)
    if np.allclose(a, b):
        return 1.0
    _, p = wilcoxon(a, b)
    return p

def print_pairwise(label_a, label_b, arr_a, arr_b, metric_label):
    p = wilcoxon_p(arr_a, arr_b)
    direction = ">" if np.mean(arr_a) > np.mean(arr_b) else "<"
    print(f"  Wilcoxon [{label_a}] {direction} [{label_b}] on {metric_label}: p = {p:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Running with {len(SEEDS)} seeds: {SEEDS}")
    print(f"Total models to train: {len(SEEDS) * len(CONFIGS)}")

    # Clean up stale checkpoints and broken model dirs
    print("\nCleaning up stale checkpoints...")
    for ckpt_dir in glob.glob("/kaggle/working/models/*/checkpoint-*"):
        try: shutil.rmtree(ckpt_dir)
        except Exception: pass

    print("Checking for broken model directories...")
    for model_dir in glob.glob("/kaggle/working/models/model_data_*"):
        if not is_model_complete(model_dir):
            print(f"  -> Incomplete: {model_dir}. Will retrain.")
            shutil.rmtree(model_dir)

    # Locate and copy dataset to writable area
    print("\nSearching for dataset files...")
    search_paths = glob.glob("/kaggle/input/**/distilbert_baseline", recursive=True)
    if not search_paths:
        raise RuntimeError("Could not find distilbert_baseline! Make sure the dataset is attached.")
    ORIGINAL_DATA_ROOT = os.path.dirname(search_paths[0])
    DATA_ROOT = "/kaggle/working/AD_COG_DATA"
    if not os.path.exists(DATA_ROOT):
        print(f"Copying data to writable dir {DATA_ROOT}...")
        shutil.copytree(ORIGINAL_DATA_ROOT, DATA_ROOT, dirs_exist_ok=True)
    print(f"Dataset ready at: {DATA_ROOT}")

    model_path = os.path.join(DATA_ROOT, "distilbert_baseline")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    print("\nLoading datasets...")
    raw_dataset = load_dataset("clinc_oos", "plus")
    datasets_map = {
        "none":            raw_dataset["train"],
        "counterfactual":  load_from_disk(os.path.join(DATA_ROOT, "high_density_dataset")),
        "generic":         load_from_disk(os.path.join(DATA_ROOT, "generic_dataset")),
    }

    # ---- Custom Trainer with weighted OOS loss ----
    class AD_COG_Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            weights = torch.ones_like(labels).float()
            if self.custom_loss_weight != 1.0:
                weights[labels == OOS_LABEL] = self.custom_loss_weight
            loss = (losses * weights).mean()
            return (loss, outputs) if return_outputs else loss

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        if "intent" in examples:
            result["labels"] = examples["intent"]
        elif "label" in examples:
            result["labels"] = examples["label"]
        return result

    eval_tokenized = raw_dataset["validation"].map(
        preprocess_function, batched=True,
        remove_columns=raw_dataset["validation"].column_names
    )
    eval_tokenized.set_format("torch")

    # ================================================================
    # TRAINING LOOP — 8 seeds × 5 configs = 40 models
    # ================================================================
    trained_count = 0
    skipped_count = 0
    for seed in SEEDS:
        for name, data_arg, loss_arg in CONFIGS:
            set_seed(seed)
            output_dir = f"./models/model_data_{data_arg}_loss_{loss_arg}_seed_{seed}"

            if is_model_complete(output_dir):
                print(f"[SKIP] {name} | Seed {seed}")
                skipped_count += 1
                continue

            print(f"\n{'='*60}")
            print(f"TRAINING: {name} | Seed {seed}")
            print(f"{'='*60}")

            train_dataset = datasets_map[data_arg]
            train_tokenized = train_dataset.map(
                preprocess_function, batched=True,
                remove_columns=train_dataset.column_names
            )
            train_tokenized.set_format("torch")

            model = DistilBertForSequenceClassification.from_pretrained(model_path)

            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=1,   # only keep 1 checkpoint to save disk
                report_to="none",
                seed=seed,
                fp16=torch.cuda.is_available()
            )

            trainer = AD_COG_Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=eval_tokenized,
            )
            trainer.custom_loss_weight = loss_arg
            trainer.train()
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Delete checkpoint subfolders immediately to preserve disk space
            for ckpt in glob.glob(f"{output_dir}/checkpoint-*"):
                try: shutil.rmtree(ckpt)
                except Exception: pass

            trained_count += 1
            print(f"[DONE] Saved to {output_dir}")

    print(f"\n\nTraining complete. Trained: {trained_count} | Skipped (already done): {skipped_count}")
    print("Starting Evaluation Phase...\n")

    # ================================================================
    # EVALUATION — full test set + failure stress-test set
    # ================================================================
    test_data = raw_dataset["test"]
    full_texts  = [x["text"] for x in test_data]
    full_labels = np.array([0 if int(x["intent"]) == OOS_LABEL else 1 for x in test_data], dtype=np.int64)

    failure_file = os.path.join(DATA_ROOT, "failure_set_top500.json")
    with open(failure_file, "r") as f:
        failure_data = json.load(f)
    fail_texts  = [s["text"] for s in failure_data]
    fail_labels = np.array([0 if int(s["label"]) == OOS_LABEL else 1 for s in failure_data], dtype=np.int64)

    # Collect per-seed metrics
    results = {
        c[0]: {"full_auroc": [], "full_fpr95": [], "fail_auroc": [], "fail_fpr95": []}
        for c in CONFIGS
    }

    for seed in SEEDS:
        for name, data_arg, loss_arg in CONFIGS:
            model_dir = f"./models/model_data_{data_arg}_loss_{loss_arg}_seed_{seed}"
            print(f"Evaluating: {name} | Seed {seed} ...", end=" ")
            full_auc,  full_fpr  = evaluate_model(model_dir, full_texts,  full_labels,  OOS_LABEL, device)
            fail_auc,  fail_fpr  = evaluate_model(model_dir, fail_texts,   fail_labels,  OOS_LABEL, device)
            results[name]["full_auroc"].append(full_auc)
            results[name]["full_fpr95"].append(full_fpr)
            results[name]["fail_auroc"].append(fail_auc)
            results[name]["fail_fpr95"].append(fail_fpr)
            print(f"Full AUROC={full_auc:.4f}  Fail AUROC={fail_auc:.4f}")

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    N = len(SEEDS)
    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS  (N={N} seeds: {SEEDS})")
    print(f"{'='*70}")
    header = f"{'Config':<28} {'FullAUROC':>12} {'FullFPR95':>12} {'FailAUROC':>12} {'FailFPR95':>12}"
    print(header)
    print("-"*70)
    for name, _, _ in CONFIGS:
        r = results[name]
        fa   = np.array(r["full_auroc"]);  ff   = np.array(r["full_fpr95"])
        ea   = np.array(r["fail_auroc"]);  ef   = np.array(r["fail_fpr95"])
        print(f"{name:<28} "
              f"{np.mean(fa):.4f}±{np.std(fa):.4f}  "
              f"{np.mean(ff):.4f}±{np.std(ff):.4f}  "
              f"{np.mean(ea):.4f}±{np.std(ea):.4f}  "
              f"{np.mean(ef):.4f}±{np.std(ef):.4f}")

    # ================================================================
    # PAIRWISE WILCOXON TESTS — one for every prose claim
    # ================================================================
    print(f"\n{'='*70}")
    print("PAIRWISE WILCOXON SIGNED-RANK TESTS")
    print(f"{'='*70}\n")

    R = results  # shorthand

    # --- FINDING 1: Loss weighting alone does nothing ---
    print("[ Finding 1 — Is 20x loss weighting helpful without better data? ]")
    print_pairwise("Baseline", "Baseline + 20x ONLY",
                   R["Baseline"]["fail_auroc"], R["Baseline + 20x ONLY"]["fail_auroc"],
                   "Fail-set AUROC")
    print_pairwise("Baseline", "Baseline + 20x ONLY",
                   R["Baseline"]["fail_fpr95"], R["Baseline + 20x ONLY"]["fail_fpr95"],
                   "Fail-set FPR95")

    # --- FINDING 2: AD-COG vs Baseline (overall benefit) ---
    print("\n[ Finding 2 — Does Full AD-COG outperform Baseline? ]")
    print_pairwise("Full AD-COG", "Baseline",
                   R["Full AD-COG"]["fail_auroc"], R["Baseline"]["fail_auroc"],
                   "Fail-set AUROC")
    print_pairwise("Full AD-COG", "Baseline",
                   R["Full AD-COG"]["fail_fpr95"], R["Baseline"]["fail_fpr95"],
                   "Fail-set FPR95")
    print_pairwise("Full AD-COG", "Baseline",
                   R["Full AD-COG"]["full_auroc"], R["Baseline"]["full_auroc"],
                   "Full-test AUROC")

    # --- FINDING 3: CF-only vs Full AD-COG (does loss weight add anything?) ---
    print("\n[ Finding 3 — Does 20x loss weighting add value ON TOP of counterfactuals? ]")
    print_pairwise("Full AD-COG", "Baseline + 40 CF ONLY",
                   R["Full AD-COG"]["fail_auroc"], R["Baseline + 40 CF ONLY"]["fail_auroc"],
                   "Fail-set AUROC")
    print_pairwise("Full AD-COG", "Baseline + 40 CF ONLY",
                   R["Full AD-COG"]["fail_fpr95"], R["Baseline + 40 CF ONLY"]["fail_fpr95"],
                   "Fail-set FPR95")
    print_pairwise("Full AD-COG", "Baseline + 40 CF ONLY",
                   R["Full AD-COG"]["full_fpr95"], R["Baseline + 40 CF ONLY"]["full_fpr95"],
                   "Full-test FPR95")

    # --- FINDING 4: Targeted vs Random (core claim) ---
    print("\n[ Finding 4 — Does targeted selection beat random augmentation (matched N=40)? ]")
    print_pairwise("Full AD-COG", "Generic Augmentation",
                   R["Full AD-COG"]["fail_auroc"], R["Generic Augmentation"]["fail_auroc"],
                   "Fail-set AUROC")
    print_pairwise("Full AD-COG", "Generic Augmentation",
                   R["Full AD-COG"]["fail_fpr95"], R["Generic Augmentation"]["fail_fpr95"],
                   "Fail-set FPR95")
    print_pairwise("Baseline + 40 CF ONLY", "Generic Augmentation",
                   R["Baseline + 40 CF ONLY"]["fail_auroc"], R["Generic Augmentation"]["fail_auroc"],
                   "Fail-set AUROC (CF-only vs Generic)")

    # --- VS ALL: AD-COG vs every other config ---
    print("\n[ Full AD-COG vs all other configs (both test sets) ]")
    for name, _, _ in CONFIGS:
        if name == "Full AD-COG":
            continue
        print_pairwise("Full AD-COG", name,
                       R["Full AD-COG"]["full_auroc"], R[name]["full_auroc"],
                       "Full-test AUROC")
        print_pairwise("Full AD-COG", name,
                       R["Full AD-COG"]["fail_auroc"], R[name]["fail_auroc"],
                       "Fail-set AUROC")

    print(f"\n{'='*70}")
    print("NOTE: Minimum achievable Wilcoxon p-value for N=8 is 0.0078")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
