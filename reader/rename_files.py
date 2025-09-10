import os

# 📂 folder to modify
folder = "/Users/gabriele/Desktop/Magistrale/Explainable_and_trustworthy_AI/progetti/venv/concept_gridlock/concepts_with_noise"
count = 0
for filename in os.listdir(folder):
    if "_240frames" in filename:
        count += 1
        new_name = filename.replace("_240frames", "")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {filename} → {new_name}")

print(f"🔍 Total files renamed: {count}")