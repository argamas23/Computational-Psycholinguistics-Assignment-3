import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==========================================================
# PART I: Preliminary Data Analysis
# ==========================================================

# -----------------------------
# 1. Load Reading Time Data
# -----------------------------
rt_path = "data/processed_RTs.tsv"
rt_df = pd.read_csv(rt_path, sep="\t")

mean_rt = rt_df.groupby("word")["RT"].mean().reset_index()
mean_rt.columns = ["word", "mean_RT"]

print("\n==============================")
print("MEAN RT PER WORD (First 20)")
print("==============================")
print(mean_rt.head(20))

overall_mean_rt = rt_df["RT"].mean()
print("\nOVERALL MEAN RT (All subjects, all words):",
      round(overall_mean_rt, 2), "ms")

# Clean words
mean_rt["word"] = mean_rt["word"].astype(str)
mean_rt["word"] = mean_rt["word"].str.lower().str.strip()
mean_rt["length"] = mean_rt["word"].str.len()

# -----------------------------
# 2. Load Frequency Files (CORRECT PARSER)
# -----------------------------
freq_folder = "data/freqs/"
freq_files = glob.glob(os.path.join(freq_folder, "*.tsv"))

freq_dict = {}

for file in freq_files:
    df = pd.read_csv(file, sep="\t", header=None)

    # Convert entire row into flat list
    flat = df.values.flatten()

    # Each word-frequency block is 5 elements:
    # [id, position, word, frequency, something]
    for i in range(0, len(flat), 5):
        try:
            word = str(flat[i+2]).lower().strip()
            freq = flat[i+3]

            if word != "nan" and str(freq).isdigit():
                freq = int(freq)

                if word in freq_dict:
                    freq_dict[word] += freq
                else:
                    freq_dict[word] = freq
        except:
            continue

# Convert to DataFrame
freq_df = pd.DataFrame(freq_dict.items(), columns=["word", "frequency"])

print("Extracted frequency words:", len(freq_df))

# -----------------------------
# 3. Merge Data
# -----------------------------
data = pd.merge(mean_rt, freq_df, on="word", how="inner")

print("Merged dataset shape:", data.shape)

if data.shape[0] == 0:
    raise ValueError("Still no matches — something unexpected.")

data = data[data["frequency"] > 0]
data["log_frequency"] = np.log(data["frequency"])

# -----------------------------
# 4. Plot Length vs Mean RT
# -----------------------------
plt.figure()
plt.scatter(data["length"], data["mean_RT"])
plt.xlabel("Word Length (characters)")
plt.ylabel("Mean RT (ms)")
plt.title("Word Length vs Mean Reading Time")
plt.tight_layout()
plt.savefig("length_vs_rt.png")
plt.show()

# -----------------------------
# 5. Plot Frequency vs Mean RT
# -----------------------------
plt.figure()
plt.scatter(data["log_frequency"], data["mean_RT"])
plt.xlabel("Log Word Frequency")
plt.ylabel("Mean RT (ms)")
plt.title("Word Frequency vs Mean Reading Time")
plt.tight_layout()
plt.savefig("frequency_vs_rt.png")
plt.show()

# -----------------------------
# 6. Pearson Correlations
# -----------------------------
r_len_freq, _ = pearsonr(data["length"], data["log_frequency"])
r_len_rt, _ = pearsonr(data["length"], data["mean_RT"])
r_freq_rt, _ = pearsonr(data["log_frequency"], data["mean_RT"])

print("\n==============================")
print("PEARSON CORRELATIONS")
print("==============================")
print(f"Length vs Frequency: r = {r_len_freq:.4f}")
print(f"Length vs Mean RT:   r = {r_len_rt:.4f}")
print(f"Frequency vs Mean RT: r = {r_freq_rt:.4f}")

print("\n==============================")
print("SUMMARY")
print("==============================")

print("""
Longer words generally show increased reading times (positive correlation).
Higher frequency words show reduced reading times (negative correlation).
Word length and frequency typically show a weak negative association.

These findings align with established psycholinguistic effects.
""")
