# phase5_retraining.py
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from datasets import load_from_disk, load_dataset

model_path = "./distilbert_baseline"

# choose which dataset to train on:
# - "./augmented_train_dataset" (only 10 hard samples)
# - "./high_density_dataset" (after adding 30 more)
train_dataset = load_from_disk("./high_density_dataset")

raw_eval = load_dataset("clinc_oos", "plus")["validation"]

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

OOS_LABEL = 42  # <-- FIX

class AD_COG_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        weights = torch.ones_like(labels).float()
        weights[labels == OOS_LABEL] = 20.0  # <-- FIX: weight correct OOS label

        loss = (losses * weights).mean()
        return (loss, outputs) if return_outputs else loss

def preprocess_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    result["labels"] = examples["intent"]
    return result

print("Preprocessing datasets...")
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
eval_tokenized = raw_eval.map(preprocess_function, batched=True, remove_columns=raw_eval.column_names)

train_tokenized.set_format("torch")
eval_tokenized.set_format("torch")

training_args = TrainingArguments(
    output_dir="./ad_cog_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = AD_COG_Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
)

print("Starting Phase 5: Robust Weighted Retraining (OOS=42)...")
trainer.train()

trainer.save_model("./ad_cog_final_model")
tokenizer.save_pretrained("./ad_cog_final_model")
print("\nSuccess! Saved to './ad_cog_final_model'")