# GenreGradient
### Multi-Class Audio-Based Genre Classification Using Spotify Acoustic Features

> Supervised learning model that predicts song genre using engineered acoustic features, dimensionality reduction, and gradient-boosted trees; built with strict leakage prevention and multi-class evaluation.

---

## 📌 Project Overview

This project builds a multi-class classification model to predict the genre of a song using Spotify-provided acoustic features.

Using 50,000 songs across 10 genres, each described by 18 predictors (continuous acoustic metrics + categorical musical attributes), the objective was to determine whether genre can be inferred purely from how a song sounds without using artist names, track titles, or linguistic metadata.

Unlike binary classification, multi-class prediction requires the model to assign the exact correct genre among ten possible categories. This stricter correctness criterion makes evaluation and preprocessing especially important.

The final model achieved:

* **Macro-Average One-vs-Rest ROC-AUC: 0.8999**

> Indicating strong discriminative performance across all genres

---

## Dataset

### Source

Data was collected using the Spotify API, consisting of 50,000 randomly sampled songs across 10 genres.

Each song contains:

* Continuous acoustic features (e.g., energy, tempo, acousticness, speechiness, loudness)
* Musical attributes (e.g., key, mode)
* String metadata (artist, track name, instance ID, retrieval date)

---

## Train/Test Split & Leakage Prevention

To ensure robust and unbiased evaluation:

* A **genre-stratified** split was performed
* For each of the 10 genres:
  * **500 songs** --> Test set
  * **4,500 songs** --> Training set
* Fixed random seed for reproducibility
* No overlap between sets

All preprocessing steps (cleaning, scaling, PCA, encoding were:
> **Fit exclusively on training data and then applied to test data**
This prevented leakage and avoided artifically inflated performance estimates.

---

## Data Cleaning & Missing Values

Missing values appeared primarily in:

* Duration
* Tempo
* Other numeric acoustic features

Markers included **-1** and **?**, which were converted to **NaN**.

Rather than imputing (which would have introduced artifical structure or certain assumptions about data distribution):

* Rows containing missing values were removed independently from train and test sets
* Dataset size remained sufficiently large after removal
* Missingness appeared random and not genre-specific

---

## Feature Engineering & Encoding Decisions

### Numeric Features

Many acoustic features (e.g., loudness, tempo, speechiness):

* Are skewed
* Contain outliers
* Not normally distributed

Instead of standard scaling, **RobustScaler** was used:

* Centers on median
* Scales by interquartile range
* Reduces influence of outliers

This ensured downstream PCA/modeling weren't dominated by extremities.

### Categorical Features

The following columns (string metadata) were removed:

* Artist name
* Track name
* Instance ID
* Retrieval date

It's possible these columns contained signal. However, the project's focus is on audio-based classification, not identity-based prediction.

#### Musical Key

> Label-encoded to retain structured categorical information.

#### Mode

> One-hot encoded.
> Avoided assigning arbitrary ordinal structure.
All categorical features were excluded from PCA and scaling to preserve interpretability.

---

## Dimensionality Reduction & Latent Structure

### PCA

To reduce noise and collinearity:

* PCA applied to scaled numeric features
* Fit only on training data
* retained **95% of variance**

### K-Means Clustering

In reduced feature space:

* Applied unsupervised K-Means clustering
* Provided latent grouping structure
* Cluster labels supplied as additional features to the classifier

This step allowed the supervised model to leverage underlying acoustic similarity patterns before genre classification.

---

## Classification Model

### Histogram-based Gradient Boosting Classifier

* Captures non-linear feature interactions
* Handles non-normal feature distributions well
* Produces calibrated probability outputs
* Scales efficiently to large datasets

### Evaluation Framework

#### One-vs-Rest (OvR) ROC-AUC
For each genre:

* A binary ROC curve was computed (genre vs. all others)
* Macro-average AUC calculated across all classes

Prediction was only considered correct when:
> The predicted genre exactly matched the true genre.

---

## Results & Visual Analysis

### (0) Multi-Class ROC Curves (OvR)

* Most curves hug the **top-left corner**, indicating high true-positive rate (TPR) at low false-positive rate (FPR)
* **Classical** and **Anime** show near-perfect separation, reflecting distinctive acoustic signatures
* **Alternative** exhibits more diagonal behavior, suggesting overalp with ROck and Pop

> The model significantly outperforms random guessing for every genre. Classification difficulty varires by acoustic overlap b/w genres.

### (1) Genre Clustering in Reduced Feature Space

* **Classical** and **Rap/Hip-Hop** occupy clearly separated regions
* **Rock, Alternative, and Electronic** exhibit heavy overlap

> Genres that are acoustically separable are easier to classify.

### (2) Audio Fingerprints Radar Chart

The radar chart summarizes how each genre deviates from the global mean across nine normalized acoustic features.
* **Classical**
  * High acousticness & instrumentalness
  * Low energy & loudness
  * Dramatic separation from other genres
* **Hip-Hop / Rap**
  * High speechiness & danceability
* **Rock & Alternative**
  * Nearly overlapping feature profiles

> Genre classification success depends on how distinctly the features encode musical conventions.

## Final Performance

**Macro-Average One-vs-Rest ROC-AUC: 0.8999**
Representing strong discriminative performance across ten genres given:

* Strict multi-class correctness
* No identity-based features
* No lyrical or textual data

The central factor behind performance is the strong alignment b/w Spotify's engineered acoustic features and genre-defining musical characteristics.
Features such as:

* Acousticness
* Speechiness
* Energy
* Instrumentalness

Directly encode structural musical properties that map cleanly to genre categories.

When combined with:

* Robust preprocessing
* Dimensionality reduction
* Latent clustering
* Non-linear gradient boosting

The model learns effective multi-class decision boundaries in high-D feature space.

## Takeaways

* Multi-class problems require stricter evaluation discipline than binary tasks
* Leakage prevention meaningfully impacts performance validity
* Acoustic feature geometry directly governs classification difficulty
* Strong preprocessing choices matter as much as model selection

## Technical Stack
* Python
* NumPy / Pandas
* scikit-learn
* RobustScaler
* PCA
* K-Means Clustering
* Histogram Gradient Boosting
* Matplotlib / Seaborn

## 📎 Acknowledgments

Pascal Wallisch, PhD.  
Data sourced from the Spotify USA Inc. API  
This project is for analytical purposes only.



