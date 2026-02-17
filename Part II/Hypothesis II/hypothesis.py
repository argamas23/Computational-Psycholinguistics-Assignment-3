import pandas as pd
import torch
import numpy as np
import os
import statsmodels.formula.api as smf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# 1. Update NLTK downloads to fix LookupError
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') # Specific fix for your error

# -----------------------------------------------------
# 2. Load Data (Adjusted for your Part II structure)
# -----------------------------------------------------
print("Loading RT and Frequency data...")
rt_path = "../../data/processed_RTs.tsv" # [cite: 10]
rt_df = pd.read_csv(rt_path, sep="\t") 

freq_list = []
freq_dir = "../../data/freqs/" # [cite: 11]
for file in os.listdir(freq_dir):
    if file.startswith("freqs-") and file.endswith(".tsv"):
        temp_df = pd.read_csv(os.path.join(freq_dir, file), sep="\t")
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
# 3. Content vs Function Classification
# -----------------------------------------------------
# Pre-tagging a unique list of words is much faster than .apply() on a large dataframe
def get_word_map(word_list):
    content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
    # Standard POS tagging for Hypothesis 2 classification 
    tagged = nltk.pos_tag(word_list)
    return {word: ('content' if tag in content_tags else 'function') for word, tag in tagged}

# -----------------------------------------------------
# 4. Processing & Surprisal Computation
# -----------------------------------------------------
rt_word_level = rt_df.groupby(['item', 'zone', 'word'])['RT'].mean().reset_index() # [cite: 13]
rt_word_level['word_length'] = rt_word_level['word'].astype(str).apply(len)
rt_word_level['log_freq'] = np.log10(rt_word_level['word'].str.lower().map(freq_dict).fillna(1))

# Batch classification
unique_words = rt_word_level['word'].astype(str).unique().tolist()
word_map = get_word_map(unique_words)
rt_word_level['word_type'] = rt_word_level['word'].astype(str).map(word_map)

print("Loading GPT-2 for Surprisal calculation...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

def compute_surprisal(words):
    surprisals = []
    context_ids = [tokenizer.eos_token_id] 
    for word in words:
        word_ids = tokenizer.encode(" " + str(word), add_special_tokens=False)
        if len(context_ids) + len(word_ids) > 1023:
            context_ids = context_ids[-512:]
        word_surprisal = 0.0
        for tid in word_ids:
            input_tensor = torch.tensor([context_ids]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            word_surprisal += -log_probs[tid].item() # -log(probability) [cite: 25]
            context_ids.append(tid)
        surprisals.append(word_surprisal)
    return surprisals

print("Computing GPT-2 surprisals per story...")
story_data = []
for _, group in tqdm(rt_word_level.groupby('item')):
    group = group.sort_values('zone')
    group['surprisal'] = compute_surprisal(group['word'].tolist())
    story_data.append(group)

final_df = pd.concat(story_data)

# -----------------------------------------------------
# 5. Hypothesis 2: Four Regression Models 
# -----------------------------------------------------
content_df = final_df[final_df['word_type'] == 'content']
function_df = final_df[final_df['word_type'] == 'function']

# Regression Models as defined in the assignment [cite: 30, 31, 32, 33]
model1 = smf.ols("RT ~ log_freq + word_length", data=content_df).fit()  # Model 1 [cite: 30]
model2 = smf.ols("RT ~ surprisal + word_length", data=content_df).fit() # Model 2 [cite: 31]
model3 = smf.ols("RT ~ log_freq + word_length", data=function_df).fit() # Model 3 [cite: 32]
model4 = smf.ols("RT ~ surprisal + word_length", data=function_df).fit()# Model 4 [cite: 33]

# -----------------------------------------------------
# 6. Visualization (3 Distinct Segments per group)
# -----------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Content Segment (Cool tones)
sns.regplot(x='log_freq', y='RT', data=content_df, ax=axes[0,0], scatter_kws={'alpha':0.2, 'color':'navy'})
axes[0,0].set_title(f"Model 1: Content (Freq)\nR-sq: {model1.rsquared:.4f}")

sns.regplot(x='surprisal', y='RT', data=content_df, ax=axes[0,1], scatter_kws={'alpha':0.2, 'color':'teal'})
axes[0,1].set_title(f"Model 2: Content (Surprisal)\nR-sq: {model2.rsquared:.4f}")

# Function Segment (Warm tones)
sns.regplot(x='log_freq', y='RT', data=function_df, ax=axes[1,0], scatter_kws={'alpha':0.2, 'color':'darkred'})
axes[1,0].set_title(f"Model 3: Function (Freq)\nR-sq: {model3.rsquared:.4f}")

sns.regplot(x='surprisal', y='RT', data=function_df, ax=axes[1,1], scatter_kws={'alpha':0.2, 'color':'orange'})
axes[1,1].set_title(f"Model 4: Function (Surprisal)\nR-sq: {model4.rsquared:.4f}")

plt.tight_layout()
plt.savefig("hypothesis2_comparison.png")

# -----------------------------------------------------
# 7. Summary Report
# -----------------------------------------------------
with open("hypothesis2_report.txt", "w") as f:
    f.write("=== HYPOTHESIS 2 RESULTS ===\n")
    f.write(f"Content Words - Freq R²: {model1.rsquared:.6f}, AIC: {model1.aic:.2f}\n")
    f.write(f"Content Words - Surp R²: {model2.rsquared:.6f}, AIC: {model2.aic:.2f}\n")
    f.write(f"Function Words - Freq R²: {model3.rsquared:.6f}, AIC: {model3.aic:.2f}\n")
    f.write(f"Function Words - Surp R²: {model4.rsquared:.6f}, AIC: {model4.aic:.2f}\n")

print("Success! Results saved in 'hypothesis2_comparison.png' and 'hypothesis2_report.txt'.")