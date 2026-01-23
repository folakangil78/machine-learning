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

ratings_path = "data.txt"
ratings_df = parse_ratings_txt(ratings_path)

print(ratings_df.head())
print(ratings_df.shape)

movies_path = "movieTitles.csv"   # movie_id, release_date, title
movies_df = pd.read_csv(
    movies_path,
    header=None,
    names=['movie_id', 'release_date', 'title']
)

movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')





