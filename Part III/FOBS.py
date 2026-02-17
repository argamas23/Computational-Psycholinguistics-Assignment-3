import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from nltk.stem import WordNetLemmatizer
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Initialize Resources
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# -----------------------------------------------------
# 1. Load Data
# -----------------------------------------------------
print("Loading data for Part III...")
rt_df = pd.read_csv("../data/processed_RTs.tsv", sep="\t")

# Load frequencies from all freqs-*.tsv files
freq_list = []
freq_dir = "../data/freqs/"
for file in os.listdir(freq_dir):
    if file.startswith("freqs-") and file.endswith(".tsv"):
        temp_df = pd.read_csv(os.path.join(freq_dir, file), sep="\t")
        if 'word' not in temp_df.columns:
            if 'item' in temp_df.columns: temp_df = temp_df.rename(columns={'item': 'word'})
            else: temp_df.columns = ['word', 'count'] + list(temp_df.columns[2:])
        freq_list.append(temp_df[['word', 'count']])

all_freqs = pd.concat(freq_list).drop_duplicates()
freq_dict = dict(zip(all_freqs['word'].astype(str).str.lower(), all_freqs['count']))

# -----------------------------------------------------
# 2. FOBS Model Construction (Lemma Bins)
# -----------------------------------------------------
# Calculate Mean RT per word
df = rt_df.groupby(['word'])['RT'].mean().reset_index()
df['word_length'] = df['word'].astype(str).apply(len)
df['word_freq'] = df['word'].str.lower().map(freq_dict).fillna(1)
df['log_word_freq'] = np.log10(df['word_freq'])

# Get Root (Lemma) and Lemma Frequency
def get_lemma(word):
    return lemmatizer.lemmatize(str(word).lower())

df['lemma'] = df['word'].apply(get_lemma)
df['lemma_length'] = df['lemma'].apply(len)
lemma_freq_map = df.groupby('lemma')['word_freq'].sum().to_dict()
df['lemma_freq'] = df['lemma'].map(lemma_freq_map)
df['log_lemma_freq'] = np.log10(df['lemma_freq'])

# -----------------------------------------------------
# 3. Hypothesis 1: Root vs Surface Frequency
# -----------------------------------------------------
model_surface = smf.ols("RT ~ log_word_freq + word_length", data=df).fit()
model_lemma = smf.ols("RT ~ log_lemma_freq + lemma_length", data=df).fit()

# -----------------------------------------------------
# 4. Hypothesis 2: Pseudo-affix Test
# -----------------------------------------------------
# Defining 5 pairs of Real vs Pseudo affixes (matched by length/freq as possible)
pseudo_words = ['finger', 'winter', 'corner', 'hammer', 'sister']
real_words = ['driver', 'worker', 'walker', 'player', 'teacher']

test_df = df[df['word'].str.lower().isin(pseudo_words + real_words)].copy()
test_df['affix_type'] = test_df['word'].str.lower().apply(
    lambda x: 'Pseudo' if x in pseudo_words else 'Real'
)

# -----------------------------------------------------
# 5. Visualization (3 Distinct Segments)
# -----------------------------------------------------
plt.figure(figsize=(18, 6))

# Segment 1: Surface Frequency Analysis (Blue)
plt.subplot(1, 3, 1)
sns.regplot(x='log_word_freq', y='RT', data=df, scatter_kws={'alpha':0.2}, color='#1f77b4', line_kws={'color':'black'})
plt.title(f"Hyp 1: Surface Model\nR-sq: {model_surface.rsquared:.4f}")
plt.xlabel("Log Word Frequency")

# Segment 2: Lemma Frequency Analysis (Green)
plt.subplot(1, 3, 2)
sns.regplot(x='log_lemma_freq', y='RT', data=df, scatter_kws={'alpha':0.2}, color='#2ca02c', line_kws={'color':'black'})
plt.title(f"Hyp 1: Root (FOBS) Model\nR-sq: {model_lemma.rsquared:.4f}")
plt.xlabel("Log Lemma Frequency")

# Segment 3: Pseudo-affix Comparison (Orange/Red)
plt.subplot(1, 3, 3)
sns.barplot(x='affix_type', y='RT', data=test_df, palette='OrRd', capsize=.1)
plt.title("Hyp 2: Pseudo vs Real Affixes")
plt.ylabel("Mean Reading Time (ms)")

plt.tight_layout()
plt.savefig("FOBS_Analysis_Results.png")

# Results Summary
print(f"\nSurface R²: {model_surface.rsquared:.4f}")
print(f"Lemma R²: {model_lemma.rsquared:.4f}")
if len(test_df) > 0:
    mean_pseudo = test_df[test_df['affix_type']=='Pseudo']['RT'].mean()
    mean_real = test_df[test_df['affix_type']=='Real']['RT'].mean()
    print(f"Mean RT Pseudo: {mean_pseudo:.2f}ms | Mean RT Real: {mean_real:.2f}ms")