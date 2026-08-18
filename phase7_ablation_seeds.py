import argparse
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from datasets import load_from_disk, load_dataset
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["none", "counterfactual", "generic"], required=True)
    parser.add_argument("--loss", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)

    model_path = "./distilbert_baseline"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Load raw baseline data
    raw_dataset = load_dataset("clinc_oos", "plus")
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"]

    OOS_LABEL = 42

    if args.data == "counterfactual":
        # Load high density dataset which has the counterfactuals merged
        train_dataset = load_from_disk("./high_density_dataset")
    elif args.data == "generic":
        # Load generic dataset
        train_dataset = load_from_disk("./generic_dataset")

    class AD_COG_Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            losses = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

            weights = torch.ones_like(labels).float()
            if args.loss != 1.0:
                weights[labels == OOS_LABEL] = args.loss

            loss = (losses * weights).mean()
            return (loss, outputs) if return_outputs else loss

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        if "intent" in examples:
            result["labels"] = examples["intent"]
        elif "label" in examples:
            result["labels"] = examples["label"]
        return result

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    train_tokenized.set_format("torch")
    eval_tokenized.set_format("torch")

    output_dir = f"./models/model_data_{args.data}_loss_{args.loss}_seed_{args.seed}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed
    )

    trainer = AD_COG_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to {output_dir}")

if __name__ == "__main__":
    main()
