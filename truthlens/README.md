# TruthLens
### Semantic Detection of Misinformation via Transformer Embeddings

> An applied study in representation learning and robust text classification, leveraging SBERT embeddings to detect misinformation in short-form social media content.

---

## 📌 Project Overview

TruthLens is a semantic classification pipeline designed to detect misinformation in tweets by embedding text into a dense, contextual vector space and applying classical, well-calibrated classifiers on top of these representations.

Rather than relying on surface-level lexical features or end-to-end fine-tuning, this project explores the effectiveness of **pretrained sentence embeddings** (SBERT) as a fixed semantic backbone, enabling strong performance, interpretability, and computational efficiency. The resulting system demonstrates high separability between misinformation and factual content while remaining robust across decision thresholds.

The final model achieves:
* **ROC-AUC:** 0.9728  
* **PR-AUC:** 0.9439  

indicating reliable performance even under class imbalance and conservative operating regimes.

---

## Dataset Context

### Tweet-Based Misinformation Corpus

The dataset consists of labeled tweets annotated as either:
* **Misinformation**, or  
* **Non-misinformation (factual / benign content)**

Key characteristics:
* Short-form, noisy text
* Informal language, abbreviations, and stylistic variance
* Moderate class imbalance, reflecting real-world prevalence rates

These properties make the task particularly well-suited for contextual embedding approaches, where meaning is derived holistically rather than token-by-token.

---

## Modeling Approach

### Sentence-Level Representation Learning

Each tweet is transformed into a fixed-length dense vector using **SBERT (all-MiniLM-L6-v2)**:
* 384-dimensional sentence embeddings
* Pretrained on semantic similarity and natural language inference objectives
* Captures contextual meaning beyond keyword overlap

This approach decouples **semantic understanding** from **classification**, allowing lightweight models to operate on a high-quality latent space.

---

### Classification Models

Two supervised classifiers were trained and compared on top of the SBERT embeddings:

| Model | Cross-Validated Score |
|------|------------------------|
| Logistic Regression | 0.9720 |
| Calibrated Linear SVM | **0.9743** |

The **calibrated SVM** was selected as the final model due to its:
* Strong generalization
* Margin-based robustness
* Well-behaved probability estimates after calibration

---

## Evaluation Metrics

Performance was evaluated on a held-out test set using threshold-independent metrics:

* **ROC-AUC:** 0.9728  
* **PR-AUC:** 0.9439  

These metrics were chosen deliberately:
* ROC-AUC evaluates global separability across thresholds
* PR-AUC emphasizes precision under class imbalance, critical for misinformation detection

---

## Results & Analysis

### Precision–Recall Curve — *High-Fidelity Retrieval*

The Precision–Recall curve maintains **near-perfect precision even beyond 80% recall**, indicating that the SBERT embeddings enable strong semantic discrimination between misinformation and factual content.

Notably:
* Precision does not collapse at high recall levels
* False positives remain minimal even as sensitivity increases

This behavior is essential in real-world deployment scenarios, where **false accusations of misinformation carry significant cost**.

---

### ROC Curve — *Latent Space Separability*

The ROC curve exhibits a steep ascent toward the top-left corner, confirming that the learned embedding space cleanly separates the two classes.

Interpretation:
* High true positive rates are achieved with negligible increases in false positives
* Performance remains stable across a wide range of thresholds

This geometry suggests the model is **robust**, not fragile to specific operating points.

---

## Design Choices & Rationale

### Why SBERT + Classical Models?

* Avoids costly end-to-end fine-tuning
* Enables faster experimentation and deployment
* Improves interpretability of downstream behavior
* Leverages strong pretrained semantic priors

### Why Calibration?

Probability calibration ensures that model outputs:
* Are meaningful as confidence estimates
* Can be thresholded reliably under different policy constraints
* Support downstream decision-making rather than raw ranking alone

---

## Key Takeaways

* Semantic embeddings can dramatically simplify text classification pipelines
* High-quality representations often matter more than model complexity
* Calibration and metric choice are essential for trustworthy deployment
* SBERT embeddings provide strong resilience to noise and stylistic variation in social media text

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
* Multi-class misinformation taxonomy
* Temporal drift analysis in narrative patterns
* Human-in-the-loop threshold optimization

---
