#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fran-pellegrino
"""

import numpy as np
import pandas as pd
import random

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
    names=['movie_id', 'release_date', 'title']
)

movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format='%Y', errors='coerce')


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
# user and movie means (only training data)
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




