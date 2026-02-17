import pandas as pd
import torch
import numpy as np
import os
import statsmodels.formula.api as smf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# 1. Load Data (RTs and Frequencies)
# -----------------------------------------------------
print("Loading RT data...")
# Loading the Reading Times file [cite: 10]
rt_df = pd.read_csv("../../data/processed_RTs.tsv", sep="\t") 

print("Loading frequency data from data/freqs/...")
freq_list = []
freq_dir = "../../data/freqs/" # Source for word frequencies [cite: 11]

for file in os.listdir(freq_dir):
    if file.startswith("freqs-") and file.endswith(".tsv"):
        temp_df = pd.read_csv(os.path.join(freq_dir, file), sep="\t")
        # Standardizing column names: look for 'word' or 'item'
        if 'word' not in temp_df.columns:
            if 'item' in temp_df.columns:
                temp_df = temp_df.rename(columns={'item': 'word'})
            else:
                cols = list(temp_df.columns)
                temp_df = temp_df.rename(columns={cols[0]: 'word', cols[1]: 'count'})
        freq_list.append(temp_df[['word', 'count']])

all_freqs = pd.concat(freq_list).drop_duplicates()
freq_dict = dict(zip(all_freqs['word'].astype(str).str.lower(), all_freqs['count']))

# -----------------------------------------------------
# 2. Preprocess: Collapse to Word Level
# -----------------------------------------------------
# Compute the average RT across all subjects [cite: 13]
rt_word_level = rt_df.groupby(['item', 'zone', 'word'])['RT'].mean().reset_index()
rt_word_level['word_length'] = rt_word_level['word'].astype(str).apply(len)

# Map frequency and calculate log frequency for Model 1 [cite: 24]
rt_word_level['word_freq'] = rt_word_level['word'].str.lower().map(freq_dict).fillna(1)
rt_word_level['log_freq'] = np.log10(rt_word_level['word_freq'])

# -----------------------------------------------------
# 3. GPT-2 Surprisal Computation (Proxy for GPT-3)
# -----------------------------------------------------
print("Loading GPT-2 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

def compute_surprisal(words):
    surprisals = []
    context_ids = [tokenizer.eos_token_id] 
    for word in words:
        word_ids = tokenizer.encode(" " + str(word), add_special_tokens=False)
        # Handle context window (max 1024)
        if len(context_ids) + len(word_ids) > 1023:
            context_ids = context_ids[-512:]
            
        word_surprisal = 0.0
        for tid in word_ids:
            input_tensor = torch.tensor([context_ids]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # Surprisal is defined as -log(probability) [cite: 25]
            word_surprisal += -log_probs[tid].item() 
            context_ids.append(tid)
        surprisals.append(word_surprisal)
    return surprisals

# -----------------------------------------------------
# 4. Run Computation & Hypothesis Testing
# -----------------------------------------------------
print("Processing stories and computing surprisal values...")
results = []
for _, group in tqdm(rt_word_level.groupby('item')):
    group = group.sort_values('zone')
    group['surprisal'] = compute_surprisal(group['word'].tolist())
    results.append(group)

final_df = pd.concat(results)

# Define Model 1: Mean RT ~ word freq + word length [cite: 24]
model1 = smf.ols("RT ~ log_freq + word_length", data=final_df).fit()

# Define Model 2: Mean RT ~ -log(gpt3 probability) + word length [cite: 25]
model2 = smf.ols("RT ~ surprisal + word_length", data=final_df).fit()

# -----------------------------------------------------
# 5. Visualization and Analysis Saving
# -----------------------------------------------------
print("\nSaving analysis results and plots...")

# Setting colors for 3 distinct segments: Frequency, Surprisal, and Comparison
plt.figure(figsize=(15, 6))

# Segment 1: Model 1 Regression
plt.subplot(1, 2, 1)
sns.regplot(x='log_freq', y='RT', data=final_df, 
            scatter_kws={'alpha':0.2, 'color':'teal'}, line_kws={'color':'red'})
plt.title(f"Model 1: RT vs Log Freq\nR-squared: {model1.rsquared:.4f}")
plt.xlabel("Log10 Frequency")
plt.ylabel("Mean RT (ms)")

# Segment 2: Model 2 Regression
plt.subplot(1, 2, 2)
sns.regplot(x='surprisal', y='RT', data=final_df, 
            scatter_kws={'alpha':0.2, 'color':'coral'}, line_kws={'color':'red'})
plt.title(f"Model 2: RT vs Surprisal\nR-squared: {model2.rsquared:.4f}")
plt.xlabel("Surprisal (-log P)")
plt.ylabel("Mean RT (ms)")

plt.tight_layout()
plt.savefig("hypothesis1_comparison.png")

# Save Detailed Summary for the PDF report [cite: 3]
with open("hypothesis1_report_summary.txt", "w") as f:
    f.write("=== HYPOTHESIS 1 COMPARISON ===\n")
    f.write(f"Model 1 R-squared: {model1.rsquared:.6f}\n")
    f.write(f"Model 2 R-squared: {model2.rsquared:.6f}\n\n")
    f.write("=== MODEL 1 DETAIL ===\n")
    f.write(model1.summary().as_text())
    f.write("\n\n=== MODEL 2 DETAIL ===\n")
    f.write(model2.summary().as_text())

print("Execution complete. Check 'hypothesis1_comparison.png' and 'hypothesis1_report_summary.txt'.")