import json

# 1. Load your uncertainty map
with open("uncertain_samples.json", "r") as f:
    uncertain_data = json.load(f)

# 2. Select the top 10 most "Confusing" samples to fix first
targets = uncertain_data[:10]

print("--- AD-COG PHASE 3: GENERATION PROMPT ---\n")
print("Copy and paste the following prompt into your LLM (GPT-4o/DeepSeek):\n")

prompt = f"""
ACT AS: A 'Linguistic Stress-Tester' (Persona-Driven Augmentation).
TASK: My small NLP model (DistilBERT) is failing on these specific sentences. 
Your goal is to generate 5 'Hard Counterfactuals' for each sample to teach the model better logic.

RULES:
1. For each sample, generate 3 'In-Distribution' variations (should belong to one of the 150 intents).
2. For each sample, generate 2 'Hard Out-of-Scope' variations (should look similar but be clearly OOS).
3. Use 'Symbolic Tagging': Label each new sentence with [LOGIC: SEMANTIC_SHIFT] or [LOGIC: DOMAIN_GAP].

SAMPLES TO FIX:
"""

for i, s in enumerate(targets):
    prompt += f"{i+1}. Text: '{s['text']}' (Model Entropy: {s['entropy']:.2f})\n"

print(prompt)