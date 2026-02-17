# Natural Stories Corpus: Word Processing Analysis

##  Overview
This project analyzes human reading times (RT) using the **Natural Stories corpus**. The goal is to explore how word length, frequency, and language model probabilities (GPT-3) predict human processing effort, alongside testing the **Frequency Ordered Bin Search (FOBS)** model of human memory.

---

## Part I: Preliminary Data Analysis

In this section, we established the baseline relationships between physical word properties and human processing speed using the `processed_RTs.tsv` and `freqs` data.

### 1. Mean Reading Time (RT) per Word
We computed the average reading time across all subjects for each word in the RT file.

* **Overall Mean RT:** 337.96 ms
* **Unique Words Matched:** 2,076 (from a pool of 3,370 extracted frequency words)

#### **Sample Mean RT (First 20 Entries)**
| Word | Mean RT (ms) | Word | Mean RT (ms) |
| :--- | :--- | :--- | :--- |
| 'Admiral | 460.22 | 'Extraordinary | 415.67 |
| 'Ah | 297.02 | 'General | 371.44 |
| 'Based | 352.65 | 'General' | 438.33 |
| 'Budge | 359.59 | 'Good | 329.59 |
| 'But | 326.30 | 'Grandmother,' | 459.87 |
| 'Come | 403.59 | 'Haven't | 396.96 |
| 'Come, | 341.78 | 'He'll | 388.56 |
| 'Do | 348.36 | 'He's | 555.67 |
| 'Don't | 343.85 | 'How | 339.75 |
| 'Except | 377.93 | 'For | 379.38 |

---

### 2. Statistical Correlations
We computed Pearson's coefficient of correlation ($r$) to quantify the relationships between word length, frequency, and mean RT.

| Relationship | Correlation ($r$) | Strength/Direction |
| :--- | :--- | :--- |
| **Length vs. Frequency** | -0.5013 | Moderate Negative |
| **Length vs. Mean RT** | 0.2003 | Weak Positive |
| **Frequency vs. Mean RT** | -0.1191 | Very Weak Negative |

---

### 3. Summary of Findings
Based on the preliminary analysis:

* **Length & RT:** There is a moderate positive relationship; longer words generally require slightly more time to process.
* **Length & Frequency:** There is a moderate negative correlation, aligning with the linguistic principle (Zipf's Law) that more frequent words tend to be shorter.
* **Frequency & RT:** Word frequency shows a very weak, statistically insignificant correlation with reading time in this dataset, suggesting it is not the strongest predictor of processing speed on its own.

---

---

## Part II: Hypothesis Testing

In this section, we conducted regression analyses to test two primary hypotheses regarding word processing predictors and word categories.

### Hypothesis 1: LM Probabilities vs. Word Frequency
**Hypothesis:** Language model probabilities (GPT-3 Surprisal) are better predictors of reading time than raw word frequency.

| Model | Formula | $R^2$ | AIC |
| :--- | :--- | :--- | :--- |
| **Model 1** | Mean RT ~ word freq + word length | 0.0105 | 711.1 |
| **Model 2** | Mean RT ~ -log(gpt3 prob) + word length | 0.0156 | 711.2 |

#### **Key Findings:**
* **Low Predictive Power:** Both models show a very low $R^2$ (~0.01), indicating that only about 1% of the variance in reading time is explained by these variables.
* **Significance:** All predictors (frequency, length, and surprisal) were non-significant ($p > 0.4$), suggesting weak individual predictive power in this sample ($N=73$).
* **Numerical Stability:** Model 1 exhibited a very high condition number ($4.67 \times 10^9$), indicating potential multicollinearity or scaling issues with raw frequency, whereas Model 2 (Surprisal) was numerically stable.

---

### Hypothesis 2: Content vs. Function Words
**Hypothesis:** Content words are processed differently than function words. We compared frequency-based models against surprisal-based models for both categories.

#### **Model Comparison Table**
| Category | Model Type | $R^2$ | AIC |
| :--- | :--- | :--- | :--- |
| **Content** | Frequency | 0.0105 | 460.9 |
| **Content** | Surprisal | 0.0428 | 459.3 |
| **Function** | Frequency | 0.0486 | 249.1 |
| **Function** | Surprisal | 0.1113 | 247.2 |

#### **Category Analysis:**
* **Content Words:** The surprisal model slightly outperformed the frequency model. This suggests that for words with high semantic weight, contextual predictability (surprisal) is a more informative predictor than raw frequency.
* **Function Words:** The difference was more pronounced here, with surprisal explaining **11.1%** of the variance compared to **4.8%** for frequency. Interestingly, the surprisal coefficient was negative, potentially reflecting the highly automated nature of function word processing.

---

### Part II Summary
The results provide preliminary support for the hypothesis that **content and function words are processed differently**. In both cases, models utilizing GPT-3 surprisal provided a better fit (lower AIC, higher $R^2$) than frequency-based models. While the small sample size ($N_{function}=27$) limited statistical significance, the trend indicates that contextual predictability is a superior metric for understanding human reading times compared to static frequency counts.

---

---

## Part III: Frequency Ordered Bin Search (FOBS)

This section explores the **Frequency Ordered Bin Search (FOBS)** model by analyzing whether human lexical access is guided more by root forms (lemmas) or surface forms, and testing the cognitive load of morphological decomposition.

### 1. Hypothesis: Root Frequency vs. Surface Frequency
**Hypothesis:** Root (lemma) frequency predicts reading times more accurately than surface frequency.

We compared two OLS regression models to determine which linguistic unit best explains the variance in Mean RT.

| Model | Predictors | $R^2$ | AIC |
| :--- | :--- | :--- | :--- |
| **Model 1 (Surface)** | Word Frequency + Word Length | 0.0081 | 711.23 |
| **Model 2 (Root)** | Lemma Frequency + Lemma Length | 0.0116 | 711.03 |

#### **Statistical Observations:**
* **Model Fit:** Model 2 (Root/Lemma) demonstrated a higher $R^2$ and a lower AIC compared to Model 1. 
* **Predictive Power:** Although the overall variance explained remains low, the improvement in Model 2 supports the hypothesis that the human mental lexicon may be organized around root forms, making lemma frequency a more potent predictor of reading time than surface frequency.

---

### 2. Hypothesis: Pseudo-Affixation and Processing Time
**Hypothesis:** Pseudo-affixed words (e.g., *finger*, where *-er* is not a morpheme) take more processing time than regularly affixed words (e.g., *driver*, where *-er* is a functional suffix).

We analyzed a selection of word pairs controlled for length and frequency to test for differences in cognitive load during morphological decomposition.

#### **Comparative Metrics:**
* **Mean RT (Pseudo-affixed):** 336.83 ms
* **Mean RT (Real-affixed):** 336.20 ms
* **T-test Results:** $t$-stat = 0.512, $p$-value = 0.622

#### **Findings:**
* **No Significant Difference:** The high $p$-value ($0.622$) indicates that there is no statistically significant difference in processing time between pseudo-affixed and regularly affixed words.
* **Psycholinguistic Implication:** The data does not support the hypothesis that pseudo-affixes increase processing difficulty. This suggests that readers may process these words via whole-word recognition or form-based "chunking" rather than a strict, cost-heavy morphological decomposition process.

---

## Final Project Summary

Across all three parts of the analysis, several key trends emerged:
1. **Predictors of Reading Time:** Contextual predictability (GPT-3 Surprisal) and Root (Lemma) frequency generally provided better model fits than raw surface frequency, though the overall effect sizes were small.
2. **Word Categories:** Content and function words exhibit distinct processing signatures, with function words showing a more pronounced relationship with LM-based surprisal ($R^2 = 0.111$).
3. **Morphological Processing:** The human processing system appears robust to pseudo-affixation, treating real and pseudo-morphemes with similar efficiency, which favors models of lexical access that prioritize root-level frequency over surface-level patterns.