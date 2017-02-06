from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os.path
import numpy as np


def get_ml_100k_dataset_simple(base_path='ml-100k'):
    """A helper function to load ml-100k database

    We only load the user_id for user data.
    We only load the movie_id for movie data.

    Args:
        base_path: The path of the ml-100k database
    Returns:
        A tuple with users, ratings and movies.
    """
    fuser = os.path.join(base_path, 'u.user')
    fdata = os.path.join(base_path, 'u.data')
    fitem = os.path.join(base_path, 'u.item')

    u_cols = ['user_id']
    users = pd.read_csv(fuser, sep='|', names=u_cols, usecols=range(1))

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(fdata, sep='\t', names=r_cols)

    m_cols = ['movie_id']
    movies = pd.read_csv(fitem, sep='|', names=m_cols, usecols=range(1))

    return users, ratings, movies


def get_sampled_ml_100k_data(
    base_path='ml-100k',
    num_movies=500,
    min_num_ratings=10,
    num_neg=20,
):
    """A helper function to sample ml-100k database and prepare training data.

    We first load the entire ml-100k database, and then sample the ratings according
    to the num_movies and num_users passed in. We can also convert the rating into
    0/1 labels and regardless of the actual ratings. Besides, the ratings will contains
    only positive data, we can fake some negative data. If in the sampled data, user
    rates less than min_num_ratings, we shall filter the user.

    Args:
        base_path: The path to ml-100k dataset.
        num_movies: Number of move to be used.
        min_num_ratings: The minimum number of rates a user need to give in the sampled data.
        num_neg: Number of negative samples generated for each user.

    Returns:
        A list of typle (user_id, [pos_movie_ids], [neg_movie_ids]).
        A dictionary contains new ids for each selected movie id.
    """
    users, ratings, movies = get_ml_100k_dataset_simple(base_path)
    # movie id selected
    movie_ids = set(np.random.choice(movies['movie_id'], num_movies, replace=False))
    # sampled ratings to be used
    sampled_ratings = ratings[ratings['movie_id'].isin(movie_ids)]
    # filter users
    user_ids = set()
    for user in users['user_id']:
        if len(sampled_ratings[sampled_ratings['user_id'] == user]) < min_num_ratings:
            continue
        user_ids.add(user)
    sampled_ratings = sampled_ratings[sampled_ratings['user_id'].isin(user_ids)]

    print("Final user selected: %d" % len(user_ids))
    print("Final ratings selected: %d" % len(sampled_ratings))
    # reindex movie data
    movie_indexer = {}
    for movie_id in movie_ids:
        movie_indexer[movie_id] = len(movie_indexer)

    data = []
    for user in user_ids:
        pos = set(sampled_ratings[sampled_ratings['user_id'] == user]['movie_id'])
        neg = set(np.random.choice(list(movie_ids - pos), num_neg, replace=False))
        data.append((user, [movie_indexer[old_id] for old_id in pos], [movie_indexer[old_id] for old_id in neg]))
    return data, movie_indexer
