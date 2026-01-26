# RateShift
### Quantifying Temporal Bias and Latent Drift in Large-Scale Rating Systems

> A project-based exploration of temporal structure and signal modeling, bias decomposition, and regularized matrix factorization using 27M Netflix movie ratings.

---

## 📌 Project Overview

This project presents a large-scale regression-based recommender system built on a substantial subset of Netflix’s **2006 Cinematch Prize dataset**, intending to model user–item interactions while explicitly identifying and analyzing **systematic temporal effects** in movie ratings.

Rather than treating time as noise, this work isolates and interprets **seasonal bias**, **rating calibration behavior**, and **content aging dynamics**, demonstrating how temporal structure influences both predictive performance and inferred user sentiment. The final model achieves an **RMSE of 0.9497393**, consistent with production-grade collaborative filtering baselines.

This repository is designed to function both as:
* a **technical report**, and
* a **self-contained case study of a large-scale recommender system**

---

## 📊 Dataset Context

### Netflix Cinematch Subset

The original Netflix Cinematch dataset contains over **100 million ratings**, collected from **400,000+ users** across **17,000+ films**. Due to computational and time constraints, this project uses a carefully curated subset consisting of:

* **27,009,840 ratings**
* **~400,000 users**
* **~5,000 movies**

This subset preserves the statistical and behavioral properties of the full dataset while enabling tractable experimentation and iterative modeling.

### Raw Data Structure

The dataset is distributed across multiple raw `.txt` and `.csv` files, including:

* **User–Movie Rating Logs**  
  Structured as repeated movie blocks followed by `(user_id, rating, timestamp)` triplets.
* **Movie Metadata File**  
  Containing movie IDs, release years, and titles, stored as comma-separated records with trailing empty fields.

All files lack explicit headers and require custom parsing logic before ingestion into analytical pipelines.

---

## ⚙️ Data Engineering & Preprocessing

### Parsing Large-Scale Text Data

Given the raw text-based structure of the dataset:
* Custom parsers were implemented to stream and process files line-by-line.
* Ratings were incrementally appended to structured tabular formats to avoid memory overflow.
* Movie metadata required explicit schema assignment due to malformed trailing delimiters.

This approach ensured correctness while maintaining scalability across ~27M observations.

---

### Missing Data Handling, Preprocessing

Missing values primarily appeared in:
* Movie metadata fields (e.g., release year)
* Sparse user–movie interactions (inherent to recommender systems)

Design choices:
* **Metadata rows with irrecoverable missing fields were dropped**, as they represented a negligible fraction of the dataset.
* **No imputation was applied to ratings**, preserving the sparsity structure critical to collaborative filtering.

This avoided introducing artificial signal while maintaining interpretability.

---

## 🧠 Model Architecture

### Regularized Matrix Factorization (Regression Formulation)

The recommender system is formulated as a **regularized regression problem**:

\[
\hat{r}_{ui} = \mu + b_u + b_i + b_t + p_u^\top q_i
\]

Where:
- \( \mu \) is the global mean rating  
- \( b_u \) and \( b_i \) are user and item bias terms  
- \( b_t \) is a **learned temporal (seasonal) bias**  
- \( p_u \) and \( q_i \) are latent factor vectors  

### Optimization

- Trained using **Stochastic Gradient Descent (SGD)**
- L2 regularization applied to all bias and latent components
- Factor dimensionality and learning rates tuned for stability and computational efficiency

---

## 🎯 Design Choices & Rationale

### Why Matrix Factorization?

- Proven effectiveness for sparse, high-dimensional interaction data
- Interpretable decomposition into biases and latent structure
- Strong empirical performance relative to computational cost

### Why Explicit Temporal Bias?

Ratings are **not exchangeable over time**:
- Seasonal sentiment fluctuations (holidays, summer releases)
- Platform usage patterns vary throughout the year

Modeling time as a learned bias term allows the system to:
- Correct systematic error
- Improve RMSE without increasing latent complexity
- Enable post-hoc interpretability of temporal effects

---

## 🚧 Key Modeling Challenges & Solutions

### 1. Non-Normal Rating Distributions
Movie ratings are discrete, bounded, and heavily skewed.

**Solution:**
- No Gaussian assumptions were imposed
- Evaluation focused on empirical calibration and RMSE
- Regularization mitigated overfitting to skewed extremes

---

### 2. Computational Constraints
Training on tens of millions of rows introduces practical limitations.

**Solution:**
- Careful factor dimensionality selection
- Single-pass SGD with optimized memory usage
- Subset-based experimentation before full training runs

---

### 3. Temporal & Seasonal Effects
User sentiment varies by time of year, independent of movie quality.

**Solution:**
- Introduced a **monthly temporal kernel** as an additive bias term
- Learned jointly during SGD optimization
- Isolated post-training for interpretability

---

## 📈 Results & Analysis

### **Final Model Performance**
- **RMSE:** **0.9497393**

This performance aligns with strong baseline collaborative filtering systems and validates the modeling approach.

---

## 📉 Seasonal Bias Line Plot — *Temporal Sentiment Oscillation*

This visualization isolates the learned temporal bias component (\( b_t \)) within the Regularized Matrix Factorization architecture, illustrating how SGD optimization adjusts predictions solely based on the month of observation.

The data exhibits a pronounced seasonal oscillation:
- **Lowest bias (most negative)** in January
- **Highest bias (least negative)** during July–August

This term effectively acts as a variable intercept, modulating the global mean independently of user preferences (\( p_u \)) or movie characteristics (\( q_i \)). The pattern confirms that the model successfully disentangled a latent seasonal signal, correcting systematic mid-year optimism and early-year rating harshness to minimize global RMSE.

---

## 📐 Calibration Curve by True Rating — *Model Reliability Diagnostics*

This diagnostic evaluates conditional bias by plotting:

\[
\mathbb{E}[\hat{r} \mid r]
\]

against true ratings.

Key findings:
- Clear **S-shaped deviation** from the identity line
- Overprediction of low ratings (1–2 stars)
- Underprediction of high ratings (4–5 stars)

This compression reflects the effect of **L2 regularization**, which penalizes extreme latent magnitudes and pulls predictions toward the global mean. The model prioritizes low-variance predictions in sparse regions, achieving best calibration around the dense mid-range (≈3 stars).

This visualization informs future improvement directions (loss functions, asymmetric penalties, factor scaling).

---

## 🎬 Ratings by Age — *Temporal Content Selection Bias*

This analysis aggregates ratings into non-uniform bins by **years since release**, revealing a non-linear relationship between content age and perceived quality.

Observed dynamics:
- Mild post-release regression (2–5 years)
- Strong monotonic increase for films older than 10 years
- Peak average rating in the 50+ year bucket (\( \mu \approx 3.88 \))

This pattern quantifies **survivorship bias**: older films that continue to be rated are disproportionately well-regarded classics. Contemporary ratings of legacy content are highly selective, filtering out mediocre films and inflating long-horizon averages.

---

## 🔍 Key Takeaways

- Temporal effects are **first-order signals**, not noise, in recommender systems
- Regularized matrix factorization can be extended to model interpretable seasonal structure
- Calibration diagnostics are critical for understanding *how* models fail, not just *how much*
- User interaction data reflects selection effects as much as preference

---

## 🚀 Future Extensions

- Time-aware latent factors (dynamic embeddings)
- Bayesian treatment of temporal uncertainty
- Asymmetric loss functions for rating extremes
- Exposure-aware modeling to correct survivorship bias

---

## 🧾 Technologies Used

- Python
- NumPy / pandas
- Custom SGD optimization
- Matplotlib / Seaborn
- Large-scale data parsing pipelines

---

## 📎 Acknowledgments
Pascal Wallisch, PhD.
Dataset sourced from the Netflix Prize Cinematch competition (2006).  
This project is for analytical purposes only.

