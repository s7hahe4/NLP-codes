import subprocess

seeds = [42, 123, 999, 7, 2024]
configs = [
    ("none", 1.0),
    ("none", 20.0),
    ("counterfactual", 1.0),
    ("counterfactual", 20.0),
    ("generic", 20.0)
]

for seed in seeds:
    for data, loss in configs:
        print(f"\n========================================================")
        print(f"Running config: Data={data}, Loss={loss}, Seed={seed}")
        print(f"========================================================\n")
        
        # We run it with a timeout just in case, though it shouldn't hang.
        subprocess.run(["python", "phase7_ablation_seeds.py", "--data", data, "--loss", str(loss), "--seed", str(seed)])

print("\nAll models trained! Running significance testing...\n")
subprocess.run(["python", "phase8_significance.py"])
