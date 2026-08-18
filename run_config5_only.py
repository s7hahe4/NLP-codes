import subprocess

seeds = [42, 123, 999, 7, 2024]

# We are only re-running Config 5
data = "generic"
loss = 20.0

print(f"========================================================")
print(f"RE-RUNNING CONFIG 5: DATA={data}, LOSS={loss} across 5 seeds")
print(f"========================================================\n")

for seed in seeds:
    print(f"\nTraining seed {seed}...")
    subprocess.run(["python", "phase7_ablation_seeds.py", "--data", data, "--loss", str(loss), "--seed", str(seed)])

print("\nConfig 5 retrained successfully! Running final significance testing...\n")
subprocess.run(["python", "phase8_significance.py"])
