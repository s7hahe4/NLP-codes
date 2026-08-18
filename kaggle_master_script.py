import os
import torch
import torch.nn as nn
import random
import numpy as np
import json
from pathlib import Path
from datasets import load_from_disk, load_dataset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon

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
    if device == "cpu":
        print("WARNING: You are running on CPU. For Kaggle, make sure to enable the T4 GPU accelerator in the notebook settings!")

    # Dynamically find the uploaded Kaggle dataset folder
    print("Searching for dataset files...")
    search_paths = glob.glob("/kaggle/input/**/distilbert_baseline", recursive=True)
    if not search_paths:
        raise RuntimeError("Could not find distilbert_baseline! Make sure the dataset is attached to the notebook.")
    
    ORIGINAL_DATA_ROOT = os.path.dirname(search_paths[0])
    
    # Kaggle's /kaggle/input is Read-Only. HuggingFace datasets will crash when trying to cache maps.
    # We must copy the data to /kaggle/working/ (which is writable) first!
    DATA_ROOT = "/kaggle/working/AD_COG_DATA"
    if not os.path.exists(DATA_ROOT):
        import shutil
        print(f"Copying data from {ORIGINAL_DATA_ROOT} to writable directory {DATA_ROOT}...")
        shutil.copytree(ORIGINAL_DATA_ROOT, DATA_ROOT, dirs_exist_ok=True)
    
    print(f"Dataset ready at: {DATA_ROOT}")
    
    OOS_LABEL = 42
    model_path = os.path.join(DATA_ROOT, "distilbert_baseline")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    raw_dataset = load_dataset("clinc_oos", "plus")
    
    datasets_map = {
        "none": raw_dataset["train"],
        "counterfactual": load_from_disk(os.path.join(DATA_ROOT, "high_density_dataset")),
        "generic": load_from_disk(os.path.join(DATA_ROOT, "generic_dataset"))
    }

    seeds = [42, 123, 999, 7, 2024]
    configs = [
        ("Baseline", "none", 1.0),
        ("Baseline + 20x ONLY", "none", 20.0),
        ("Baseline + 40 CF ONLY", "counterfactual", 1.0),
        ("Full AD-COG", "counterfactual", 20.0),
        ("Generic Augmentation", "generic", 20.0)
    ]

    class AD_COG_Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            weights = torch.ones_like(labels).float()
            # In Trainer, self.args holds our custom training args but we can just use an attribute
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

    eval_tokenized = raw_dataset["validation"].map(preprocess_function, batched=True, remove_columns=raw_dataset["validation"].column_names)
    eval_tokenized.set_format("torch")

    # Train all 25 models
    for seed in seeds:
        for name, data_arg, loss_arg in configs:
            set_seed(seed)
            output_dir = f"./models/model_data_{data_arg}_loss_{loss_arg}_seed_{seed}"
            
            if os.path.exists(output_dir):
                print(f"Skipping {name} Seed {seed} - already trained.")
                continue

            print(f"\n========================================================")
            print(f"TRAINING: {name} (Seed {seed})")
            print(f"========================================================")
            
            train_dataset = datasets_map[data_arg]
            train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
            train_tokenized.set_format("torch")

            # Re-init model
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
                report_to="none",
                seed=seed,
                fp16=torch.cuda.is_available() # Fast mixed precision on GPU
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

    print("\n\nAll models trained! Starting Evaluation Phase...\n")

    # Evaluate
    test_data = raw_dataset["test"]
    full_texts = [x["text"] for x in test_data]
    full_labels = np.array([0 if int(x["intent"]) == OOS_LABEL else 1 for x in test_data], dtype=np.int64)

    failure_file = os.path.join(DATA_ROOT, "failure_set_top500.json")
    with open(failure_file, "r") as f:
        failure_data = json.load(f)
    failure_texts = [s["text"] for s in failure_data]
    failure_labels = np.array([0 if int(s["label"]) == OOS_LABEL else 1 for s in failure_data], dtype=np.int64)

    results = {c[0]: {"full_auroc": [], "full_fpr95": [], "fail_auroc": [], "fail_fpr95": []} for c in configs}

    for seed in seeds:
        for name, data_arg, loss_arg in configs:
            model_dir = f"./models/model_data_{data_arg}_loss_{loss_arg}_seed_{seed}"
            full_auc, full_fpr = evaluate_model(model_dir, full_texts, full_labels, OOS_LABEL, device)
            fail_auc, fail_fpr = evaluate_model(model_dir, failure_texts, failure_labels, OOS_LABEL, device)
            
            results[name]["full_auroc"].append(full_auc)
            results[name]["full_fpr95"].append(full_fpr)
            results[name]["fail_auroc"].append(fail_auc)
            results[name]["fail_fpr95"].append(fail_fpr)

    print("\n" + "="*50)
    print("FINAL SIGNIFICANCE TESTING RESULTS")
    print("="*50)
    
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
            _, p_full = wilcoxon(ad_cog_full_auroc, full_auc_arr)
            _, p_fail = wilcoxon(ad_cog_fail_auroc, fail_auc_arr)
            print(f"  -> p-value vs Full AD-COG (Full Test): {p_full:.4f}")
            print(f"  -> p-value vs Full AD-COG (Fail Test): {p_fail:.4f}")

if __name__ == "__main__":
    main()
