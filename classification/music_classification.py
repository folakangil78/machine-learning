#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fran-pellegrino
"""

import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

random.seed(13639406)

# Read the CSV
df = pd.read_csv("musicData1.csv")

# Train / Test Split

# Column name for the target
TARGET_COL = "music_genre"

train_parts = []
test_parts = []

# For reproducibility across pandas sampling
rng = np.random.default_rng(13639406)

for genre, genre_df in df.groupby(TARGET_COL):
    # Shuffle indices deterministically
    shuffled_idx = rng.permutation(len(genre_df))
    
    test_idx = shuffled_idx[:500]
    train_idx = shuffled_idx[500:]
    
    test_parts.append(genre_df.iloc[test_idx])
    train_parts.append(genre_df.iloc[train_idx])

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df  = pd.concat(test_parts).reset_index(drop=True)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# Replace known missing-value markers with NaN
# Handle Missing Values (after split)
MISSING_MARKERS = ["?", -1]

train_df = train_df.replace(MISSING_MARKERS, np.nan)
test_df  = test_df.replace(MISSING_MARKERS, np.nan)

# Drop rows with any missing values
train_df = train_df.dropna().reset_index(drop=True)
test_df  = test_df.dropna().reset_index(drop=True)

print("Train size after dropping missing:", train_df.shape)
print("Test size after dropping missing:", test_df.shape)

# Separate Features and Target

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# Encode String & Categorical Variables

genre_encoder = LabelEncoder()
y_train_enc = genre_encoder.fit_transform(y_train)
y_test_enc  = genre_encoder.transform(y_test)

key_encoder = LabelEncoder()

X_train["key"] = key_encoder.fit_transform(X_train["key"])
X_test["key"]  = key_encoder.transform(X_test["key"])

categorical_cols = ["mode"]

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

train_cat = ohe.fit_transform(X_train[categorical_cols])
test_cat  = ohe.transform(X_test[categorical_cols])

cat_feature_names = ohe.get_feature_names_out(categorical_cols)

train_cat_df = pd.DataFrame(train_cat, columns=cat_feature_names, index=X_train.index)
test_cat_df  = pd.DataFrame(test_cat, columns=cat_feature_names, index=X_test.index)

# Drop original categorical columns and append encoded ones
X_train = X_train.drop(columns=categorical_cols)
X_test  = X_test.drop(columns=categorical_cols)

X_train = pd.concat([X_train, train_cat_df], axis=1)
X_test  = pd.concat([X_test, test_cat_df], axis=1)

print("Final training feature shape:", X_train.shape)
print("Final test feature shape:", X_test.shape)

print("Encoded genre classes:")
for i, g in enumerate(genre_encoder.classes_):
    print(i, "→", g)

# train test split and preprocessing done

# Identify one-hot columns, Separate Numeric vs Categorical Features

# Drop non-numeric / identifier / text columns
TEXT_ID_COLS = [
    "instance_id",
    "artist_name",
    "track_name",
    "obtained_date"
]

X_train = X_train.drop(columns=[c for c in TEXT_ID_COLS if c in X_train.columns])
X_test  = X_test.drop(columns=[c for c in TEXT_ID_COLS if c in X_test.columns])

categorical_ohe_cols = [c for c in X_train.columns if c.startswith("mode_")]

numeric_cols = [c for c in X_train.columns if c not in categorical_ohe_cols]

# Scale Numeric Features (Train-only Fit)

scaler = RobustScaler()

X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_num_scaled  = scaler.transform(X_test[numeric_cols])

X_train_scaled = np.hstack([
    X_train_num_scaled,
    X_train[categorical_ohe_cols].values
])

X_test_scaled = np.hstack([
    X_test_num_scaled,
    X_test[categorical_ohe_cols].values
])

# Dimensionality Reduction (PCA)

pca = PCA(n_components=0.95, random_state=13639406)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("Original dim:", X_train_scaled.shape[1])
print("Reduced dim:", X_train_pca.shape[1])

# Clustering in Reduced Space (Unsupervised)

kmeans = KMeans(n_clusters=10, random_state=13639406, n_init=20)

train_clusters = kmeans.fit_predict(X_train_pca)
test_clusters  = kmeans.predict(X_test_pca)

X_train_final = np.hstack([X_train_pca, train_clusters.reshape(-1, 1)])
X_test_final  = np.hstack([X_test_pca, test_clusters.reshape(-1, 1)])

# Multi-Class Classification Model

clf = HistGradientBoostingClassifier(
    max_depth=8,
    learning_rate=0.05,
    max_iter=300,
    random_state=13639406
)

clf.fit(X_train_final, y_train_enc)

# Multi-Class ROC-AUC Evaluation

y_test_proba = clf.predict_proba(X_test_final)

auc_ovr = roc_auc_score(
    y_test_enc,
    y_test_proba,
    multi_class="ovr",
    average="macro"
)

print(f"Macro-average One-vs-Rest AUC: {auc_ovr:.4f}")

# ROC Curve Plot (Multi-Class)

class_to_genre = {
    i: genre
    for i, genre in enumerate(genre_encoder.classes_)
}

n_classes = len(np.unique(y_test_enc))
y_test_bin = label_binarize(y_test_enc, classes=range(n_classes))

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
    plt.plot(fpr, tpr, alpha=0.5, label=class_to_genre[i])

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curves (OvR)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# Genre Clustering Visualization (2D PCA)

pca_2d = PCA(n_components=2, random_state=13639406)

X_vis = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=y_train_enc,
    cmap="tab10",
    alpha=0.6,
    s=10
)

handles, _ = scatter.legend_elements()
labels = [class_to_genre[i] for i in range(n_classes)]

plt.legend(
    handles=handles,
    labels=labels,
    title="Genre",
    fontsize=8
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Genre Clustering in Reduced Feature Space")
plt.tight_layout()
plt.show()

# Extra Credit Visualization/Insight

fingerprint_features = [
    "energy",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "valence",
    "tempo",
    "loudness",
    "liveness"
]

# Build analysis dataframe (train only)
analysis_df = X_train.copy()
analysis_df["genre"] = y_train_enc

# Keep only fingerprint features
analysis_df = analysis_df[fingerprint_features + ["genre"]]


scaler_fp = StandardScaler()

analysis_df[fingerprint_features] = scaler_fp.fit_transform(
    analysis_df[fingerprint_features]
)

genre_fingerprints = (
    analysis_df
    .groupby("genre")[fingerprint_features]
    .mean()
)

genre_fingerprints.index = [
    genre_encoder.classes_[i] for i in genre_fingerprints.index
]

def radar_plot(df, title):
    labels = df.columns
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(9, 9),
        subplot_kw=dict(polar=True)
    )

    for genre, values in df.iterrows():
        vals = values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=genre)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels
    )

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8
    )

    plt.tight_layout()
    plt.show()
    
radar_plot(
    genre_fingerprints,
    title="Genre Audio Fingerprints (Normalized Feature Profiles)"
)

