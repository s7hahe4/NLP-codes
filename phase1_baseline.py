import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the CLINC150 Dataset
dataset = load_dataset("clinc_oos", "plus")

# 2. Initialize the DistilBERT Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# 3. Tokenize the Data and Rename Label Column
def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(examples["text"], padding="max_length", truncation=True)
    # The Trainer looks for a column named 'labels' to calculate loss
    result["labels"] = examples["intent"] 
    return result

# Map the function and remove old columns to save memory on your laptop
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text", "intent"]
)

# 4. Prepare Model (150 In-Distribution classes)
num_labels = 151
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 5. Define Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 6. Set Training Arguments (Using eval_strategy for latest versions)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="epoch",         
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"                 # Prevents unnecessary cloud logging
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

# 8. Start Training
print("Starting Phase 1: Baseline Training with Corrected Labels...")
trainer.train()

# 9. Final Evaluation
results = trainer.evaluate()
print(f"Baseline Results: {results}")

# 10. Save the Model for Phase 2 (Active Discovery)
model.save_pretrained("./distilbert_baseline")
tokenizer.save_pretrained("./distilbert_baseline")