import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. LOAD MEAN READING TIME DATA
# -------------------------------------------------

rt_path = "data/processed_RTs.tsv"

rt_df = pd.read_csv(rt_path, sep="\t")

# Clean column names (remove whitespace)
rt_df.columns = rt_df.columns.str.strip()

print("Columns in processed_RTs.tsv:", rt_df.columns)

# Auto-detect correct columns
if "word" not in rt_df.columns:
    # assume first column is word
    rt_df.rename(columns={rt_df.columns[0]: "word"}, inplace=True)

if "mean_RT" not in rt_df.columns:
    # assume second column is mean_RT
    rt_df.rename(columns={rt_df.columns[1]: "mean_RT"}, inplace=True)

# Final check
if "word" not in rt_df.columns or "mean_RT" not in rt_df.columns:
    raise ValueError("Could not identify word and mean_RT columns.")

rt_df["word"] = rt_df["word"].astype(str).str.strip().str.lower()


# -------------------------------------------------
# 2. LOAD FREQUENCY FILES
# -------------------------------------------------

freq_files = glob.glob("data/freqs/freqs-*.tsv")

freq_list = []

for file in freq_files:
    df = pd.read_csv(file, sep="\t", header=None)

    # Based on your earlier output:
    # Column 2 = word
    # Column 3 = frequency
    word_col = df.columns[2]
    freq_col = df.columns[3]

    temp = df[[word_col, freq_col]].copy()
    temp.columns = ["word", "frequency"]

    freq_list.append(temp)

freq_df = pd.concat(freq_list, ignore_index=True)

freq_df["word"] = freq_df["word"].astype(str).str.strip().str.lower()
freq_df["frequency"] = pd.to_numeric(freq_df["frequency"], errors="coerce")

# Sum frequencies across files
freq_df = freq_df.groupby("word", as_index=False)["frequency"].sum()

print("Frequency data loaded successfully.")

# -------------------------------------------------
# 3. MERGE RT + FREQUENCY
# -------------------------------------------------

df = rt_df.merge(freq_df, on="word", how="inner")

# Drop missing
df = df.dropna()

# -------------------------------------------------
# 4. CREATE PREDICTORS
# -------------------------------------------------

# Word length
df["word_length"] = df["word"].str.len()

# Model 1 predictor: log frequency
df["log_freq"] = np.log(df["frequency"] + 1)

# Compute probability directly from frequency
total_tokens = df["frequency"].sum()
df["probability"] = df["frequency"] / total_tokens

# Model 2 predictor: surprisal
df["surprisal"] = -np.log(df["probability"] + 1e-12)

# -------------------------------------------------
# 5. MODEL 1: RT ~ log_freq + word_length
# -------------------------------------------------

X1 = df[["log_freq", "word_length"]]
X1 = sm.add_constant(X1)
y = df["mean_RT"]

model1 = sm.OLS(y, X1).fit()

print("\n==============================")
print("MODEL 1: RT ~ log_freq + length")
print("==============================")
print(model1.summary())

# -------------------------------------------------
# 6. MODEL 2: RT ~ surprisal + word_length
# -------------------------------------------------

X2 = df[["surprisal", "word_length"]]
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit()

print("\n==============================")
print("MODEL 2: RT ~ surprisal + length")
print("==============================")
print(model2.summary())

# -------------------------------------------------
# 7. COMPARE R²
# -------------------------------------------------

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(f"Model 1 R²: {model1.rsquared:.4f}")
print(f"Model 2 R²: {model2.rsquared:.4f}")

if model1.rsquared > model2.rsquared:
    print("→ Model 1 fits better.")
elif model2.rsquared > model1.rsquared:
    print("→ Model 2 fits better.")
else:
    print("→ Both models fit equally well.")

# -------------------------------------------------
# 8. VISUALIZATIONS
# -------------------------------------------------

plt.figure()
plt.scatter(df["log_freq"], df["mean_RT"])
plt.xlabel("Log Frequency")
plt.ylabel("Mean Reading Time")
plt.title("Mean RT vs Log Frequency")
plt.savefig("model1_logfreq_vs_rt.png")
plt.close()

plt.figure()
plt.scatter(df["surprisal"], df["mean_RT"])
plt.xlabel("Surprisal (-log probability)")
plt.ylabel("Mean Reading Time")
plt.title("Mean RT vs Surprisal")
plt.savefig("model2_surprisal_vs_rt.png")
plt.close()

print("\nPlots saved:")
print(" - model1_logfreq_vs_rt.png")
print(" - model2_surprisal_vs_rt.png")
