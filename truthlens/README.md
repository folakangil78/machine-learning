# TruthLens
### Semantic Detection of Misinformation via Transformer Embeddings

> An applied study in representation learning and robust text classification, leveraging SBERT embeddings to detect misinformation in short-form social media content (tweets).

---

## 📌 Project Overview

TruthLens is a semantic classification pipeline designed to detect misinformation in tweets by embedding text into a dense, contextual vector space and applying classical, well-calibrated classifiers on top of these representations.

Rather than relying on surface-level lexical features or end-to-end fine-tuning, this project explores the effectiveness of **pretrained sentence embeddings** (SBERT) as a fixed semantic backbone, enabling strong performance, interpretability, and computational efficiency. The resulting system demonstrates high separability between misinformation and factual content while remaining robust across decision thresholds.

The final model achieves:
* **ROC-AUC:** 0.9728  
* **PR-AUC:** 0.9439  

---

## Dataset Context

### Tweet-Based Misinformation Corpus

The dataset (sourced from HuggingFace) consists of pre-labeled tweets annotated as either:
* **Misinformation**, or  
* **Non-misinformation (factual / benign content)**

Key characteristics:
* Short-form, noisy text
* Informal language, abbreviations, and stylistic variance
* Moderate class imbalance, reflecting real-world prevalence rates

These properties make the task particularly well-suited for holistic, contextual embedding approaches rather than token-by-token.

---

## Modeling Approach

### Sentence-Level Representation Learning

Each tweet is transformed into a dense, fixed-length vector using **SBERT (all-MiniLM-L6-v2)**:
* 384-dimensional sentence embeddings
* Pretrained on semantic similarity and natural language inference
* Captures contextual meaning beyond keyword overlap

The aim is to decouple **semantic understanding** from **classification**, allowing lightweight models to operate on a high-quality latent space.

---

### Classification Models

Two supervised classifiers were trained and compared on top of the SBERT embeddings:

| Model | Cross-Validated Score |
|------|------------------------|
| Logistic Regression | 0.9720 |
| Calibrated Linear SVM | **0.9743** |

The **calibrated SVM** yielded:
* Stronger generalization
* Margin-based robustness
* Better probability estimates post calibration

---

## Evaluation Metrics

Performance was evaluated on a separate test set:

* **ROC-AUC:** 0.9728 | evaluates global separability across thresholds
* **PR-AUC:** 0.9439 | emphasizes precision under class imbalance, which is likely to exist with misinf. detection

---

## Results & Analysis

### Precision–Recall Curve

The PR curve maintains **near-perfect precision even beyond 80% recall**, implying that the SBERT embeddings enable strong semantic discrimination b/w misin. and fact.

* Precision does not collapse at high recall levels
* False positives remain minimal even as sensitivity goes up

This is pertinent because *false accusations of something being misinf. can exact significant costs*.

---

### ROC Curve

There is a steep ascent toward the top-left corner, confirming that the learned embedding space cleanly separates the two classes.

* High true positive rates are achieved with negligible increases in false positives
* Performance is stable across the range of thresholds

---

## Design Choices & Rationale

### SBERT & Classical Models

* No costly end-to-end fine-tuning
* Faster experimentation and deployment
* Improved interpretability of downstream behavior
* Leverages strong pretrained semantic priors

### Calibration

Probability calibration ensures that model outputs:
* Are meaningful as confidence estimates
* Can be thresholded reliably under different policy constraints

---

## Key Takeaways

* Semantic embeddings dramatically simplify text classification pipelines
* Calibration and metric choice are essential for trustworthy deployment
* SBERT embeddings are resilient to noise and textual variation in social media text

---

## Technologies Used

* Python
* Sentence-BERT (all-MiniLM-L6-v2)
* scikit-learn
* Logistic Regression
* Linear SVM with probability calibration
* NumPy / pandas
* Matplotlib

---

## Future Extensions

* Domain-adaptive embedding fine-tuning
* Temporal drift analysis in narrative patterns

---
