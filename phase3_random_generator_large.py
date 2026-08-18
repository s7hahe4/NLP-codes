import random
from datasets import load_dataset

# 1. Load the CLINC150 Dataset train split
dataset = load_dataset("clinc_oos", "plus")["train"]

# 2. Select 40 random samples across the full intent space
random.seed(123)  # Different seed than last time just to be safe
indices = random.sample(range(len(dataset)), 40)
targets = [dataset[i] for i in indices]

print("=== AD-COG PHASE 3: RANDOM GENERATION PROMPT (GENERIC BASELINE LARGE N=40) ===\n")
print("To avoid overwhelming the LLM, here are 4 separate prompt blocks.")
print("Paste each block into a new message in the same chat and collect the JSON.\n")

prompt_header = """ACT AS: A 'Linguistic Stress-Tester' (Persona-Driven Augmentation).
TASK: My small NLP model (DistilBERT) needs to learn better logic on these specific sentences. 
Your goal is to generate 5 'Hard Counterfactuals' for each sample to teach the model better logic.

RULES:
1. For each sample, generate 3 'In-Distribution' variations (should belong to one of the 150 intents).
2. For each sample, generate 2 'Hard Out-of-Scope' variations (should look similar but be clearly OOS).
3. Use 'Symbolic Tagging': Label each new sentence with [LOGIC: SEMANTIC_SHIFT] or [LOGIC: DOMAIN_GAP].
4. Output your response as a valid JSON array of objects with the keys: "text", "label" (the integer label index), and "logic". Do not output anything other than the JSON.

SAMPLES TO FIX:
"""

for block in range(4):
    print(f"\n--- PROMPT BLOCK {block + 1} / 4 ---\n")
    print(prompt_header)
    for i in range(10):
        s = targets[block * 10 + i]
        print(f"{i+1}. Text: '{s['text']}' (Label Index: {s['intent']})")
