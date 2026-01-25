#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fran-pellegrino
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Reproducibility
SEED = 13639406
np.random.seed(SEED)
random.seed(SEED)

# df creation from txt file
def parse_ratings_txt(filepath):
    """
    Parses Netflix-style ratings file into a DataFrame with columns:
    [movie_id, user_id, rating, date]
    """
    data = []
    current_movie_id = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Movie ID line
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                user_id, rating, date = line.split(',')
                data.append((
                    current_movie_id,
                    int(user_id),
                    int(rating),
                    date
                ))

    df = pd.DataFrame(
        data,
        columns=['movie_id', 'user_id', 'rating', 'date']
    )

    df['date'] = pd.to_datetime(df['date'])
    return df

# file read in
ratings_path = "dataSet/data.txt"
ratings_df = parse_ratings_txt(ratings_path)

print(ratings_df.head())
print(ratings_df.shape)

movies_path = "dataSet/movieTitles.csv"   # movie_id, release_date, title
movies_df = pd.read_csv(
    movies_path,
    header=None,
    names=[
        "movie_id",
        "release_year",
        "title",
        "junk1",
        "junk2" # two empty fields for whatever reason, corrupted csv
    ],
    engine="python"
)

print(movies_df.head())
print()
print(movies_df.dtypes)
print()
print(movies_df.isna().mean())
print()

# Drop the junk columns
movies_df = movies_df.drop(columns=["junk1", "junk2"])

movies_df = movies_df.reset_index(drop=True)

# Enforce correct dtypes
movies_df['movie_id'] = pd.to_numeric(
    movies_df['movie_id'],
    errors='coerce'
)

movies_df['release_year'] = pd.to_numeric(
    movies_df['release_year'],
    errors='coerce'
)

movies_df = movies_df.dropna(subset=['movie_id', 'release_year'])
movies_df['movie_id'] = movies_df['movie_id'].astype(int)
movies_df['release_year'] = movies_df['release_year'].astype(int)

# Convert release year → datetime (Jan 1 of that year)
movies_df['release_date'] = pd.to_datetime(
    movies_df['release_year'],
    format='%Y',
)

# redundant col cleanup
movies_df = movies_df.drop(columns=['release_year'])

print(movies_df.head())
print()
print(movies_df.dtypes)
print()
print(movies_df.isna().mean())
print()

# test split fxn
def leave_one_out_by_movie(df, seed=SEED):
    """
    For each movie_id:
      - randomly select ONE rating → test set
      - all others → training set
    """
    rng = np.random.default_rng(seed)

    test_indices = []

    for movie_id, group in df.groupby('movie_id'):
        # Choose one random row index for this movie
        chosen_idx = rng.choice(group.index)
        test_indices.append(chosen_idx)

    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.drop(index=test_indices).reset_index(drop=True)

    return train_df, test_df

# execute split
train_df, test_df = leave_one_out_by_movie(ratings_df)

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)
print("Unique movies in test:", test_df['movie_id'].nunique())

'''
    preprocessing after split to prevent leakage
    keeping data in long format, allowing data-absence to carry signal
    normalize ratings and remove user/movie bias
    user not rating a movie is meaningful, so not replacing with 0s
    preserves user taste, movie popularity, selective exposure behavior
'''
# user and movie means (only training data) IMPUTATION BEGIN HERE
user_mean = train_df.groupby('user_id')['rating'].mean()
movie_mean = train_df.groupby('movie_id')['rating'].mean()
global_mean = train_df['rating'].mean()

def normalize(row):
    u = row['user_id']
    m = row['movie_id']
    return (
        row['rating']
        - user_mean.get(u, global_mean)
        - movie_mean.get(m, global_mean)
        + global_mean
    )

train_df['rating_norm'] = train_df.apply(normalize, axis=1)

# Add temporal features, discretize into month of year buckets for signal
for df in [train_df, test_df]:
    df['month'] = df['date'].dt.month
    
# Reindex users and movies
user_ids = train_df['user_id'].unique()
movie_ids = train_df['movie_id'].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
movie_map = {m: i for i, m in enumerate(movie_ids)}

n_users = len(user_map)
n_movies = len(movie_map)
n_factors = 30   # sweet spot for speed vs accuracy

# training loop and model init

# hyperparams
lr = 0.01
reg = 0.05
epochs = 10

# Global mean
mu = train_df['rating'].mean()

# Parameters
bu = np.zeros(n_users)
bm = np.zeros(n_movies)
bm_month = np.zeros((n_movies, 12))

P = 0.1 * np.random.randn(n_users, n_factors)
Q = 0.1 * np.random.randn(n_movies, n_factors)

for epoch in range(epochs):
    train_df = train_df.sample(frac=1.0, random_state=SEED + epoch)

    for row in train_df.itertuples():
        u = user_map[row.user_id]
        m = movie_map[row.movie_id]
        month = row.month - 1
        r = row.rating

        pred = (
            mu
            + bu[u]
            + bm[m]
            + bm_month[m, month]
            + np.dot(P[u], Q[m])
        )

        err = r - pred

        # Updates
        bu[u] += lr * (err - reg * bu[u])
        bm[m] += lr * (err - reg * bm[m])
        bm_month[m, month] += lr * (err - reg * bm_month[m, month])

        P[u] += lr * (err * Q[m] - reg * P[u])
        Q[m] += lr * (err * P[u] - reg * Q[m])

    print(f"Epoch {epoch+1}/{epochs} complete")
    
# prediction and rmse on test set
def predict(row):
    u = user_map.get(row.user_id)
    m = movie_map.get(row.movie_id)
    month = row.month - 1

    if u is None or m is None:
        return mu

    return (
        mu
        + bu[u]
        + bm[m]
        + bm_month[m, month]
        + np.dot(P[u], Q[m])
    )

test_df['pred'] = test_df.apply(predict, axis=1)

rmse = np.sqrt(np.mean((test_df['rating'] - test_df['pred']) ** 2))
print("Test RMSE:", rmse)

# vis 1 seasonal bias line plot
avg_seasonal_bias = bm_month.mean(axis=0)

plt.figure(figsize=(10,4))
plt.plot(range(1,13), avg_seasonal_bias, marker='o')
plt.xticks(range(1,13))
plt.xlabel("Month")
plt.ylabel("Average Seasonal Movie Bias")
plt.title("Seasonal Effect in Movie Ratings Learned by Model")
plt.grid(True)
plt.show()

# vis 2: calibration curve by true rating (model reliability plot)

# Bin by true rating (1–5 stars)
calibration = (
    test_df
    .groupby('rating')
    .agg(
        mean_prediction=('pred', 'mean'),
        count=('pred', 'size')
    )
    .reset_index()
)

plt.figure(figsize=(7,6))

# Perfect calibration reference
plt.plot(
    [1, 5],
    [1, 5],
    linestyle='--',
    color='gray',
    label='Perfect calibration'
)

# Model calibration curve
plt.plot(
    calibration['rating'],
    calibration['mean_prediction'],
    marker='o',
    linewidth=2,
    label='Model prediction'
)

# Annotate sample sizes
for _, row in calibration.iterrows():
    plt.text(
        row['rating'],
        row['mean_prediction'] + 0.05,
        f"n={int(row['count'])}",
        ha='center',
        fontsize=9
    )

plt.xticks([1, 2, 3, 4, 5])
plt.yticks([1, 2, 3, 4, 5])
plt.xlabel("True Rating")
plt.ylabel("Average Predicted Rating")
plt.title("Model Calibration by True Rating")
plt.legend()
plt.grid(True)
plt.show()





