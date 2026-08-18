import random
from datasets import load_dataset

# 1. Load the CLINC150 Dataset train split
dataset = load_dataset("clinc_oos", "plus")["train"]

# 2. Select 10 random samples across the full intent space
random.seed(42)  # For reproducibility of the prompt
indices = random.sample(range(len(dataset)), 10)
targets = [dataset[i] for i in indices]

print("--- AD-COG PHASE 3: RANDOM GENERATION PROMPT (GENERIC BASELINE) ---\n")
print("Copy and paste the following prompt into your LLM (GPT-4o/DeepSeek):\n")

prompt = f"""
ACT AS: A 'Linguistic Stress-Tester' (Persona-Driven Augmentation).
TASK: My small NLP model (DistilBERT) needs to learn better logic on these specific sentences. 
Your goal is to generate 5 'Hard Counterfactuals' for each sample to teach the model better logic.

RULES:
1. For each sample, generate 3 'In-Distribution' variations (should belong to one of the 150 intents).
2. For each sample, generate 2 'Hard Out-of-Scope' variations (should look similar but be clearly OOS).
3. Use 'Symbolic Tagging': Label each new sentence with [LOGIC: SEMANTIC_SHIFT] or [LOGIC: DOMAIN_GAP].

SAMPLES TO FIX:
"""

for i, s in enumerate(targets):
    prompt += f"{i+1}. Text: '{s['text']}' (Label Index: {s['intent']})\n"

print(prompt)
