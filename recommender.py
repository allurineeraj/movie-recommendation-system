import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    movies['genres'] = movies['genres'].replace(
        '(no genres listed)', '')
    return movies, ratings

def build_content_model(movies):
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(
        movies.index,
        index=movies['title']).drop_duplicates()
    return cosine_sim, indices

def content_recommend(title, movies, cosine_sim,
                      indices, n=10):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1], reverse=True)[1:n+1]
    movie_idx = [i[0] for i in sim_scores]
    scores    = [round(i[1], 3) for i in sim_scores]
    result = movies[['title','genres']].iloc[movie_idx].copy()
    result['similarity_score'] = scores
    return result.reset_index(drop=True)

def build_collab_model(ratings):
    matrix = ratings.pivot_table(
        index='userId', columns='movieId', values='rating')
    filled = matrix.fillna(0)
    user_sim = cosine_similarity(filled)
    user_sim_df = pd.DataFrame(
        user_sim,
        index=matrix.index,
        columns=matrix.index)
    return matrix, user_sim_df

def collab_recommend(user_id, movies, matrix,
                     user_sim_df, n=10):
    if user_id not in user_sim_df.index:
        return pd.DataFrame()
    sim_users  = user_sim_df[user_id].sort_values(
        ascending=False)[1:11].index
    sim_ratings = matrix.loc[sim_users]
    rated = matrix.loc[user_id].dropna().index
    unrated = sim_ratings.drop(
        columns=rated, errors='ignore')
    avg_scores = unrated.mean(
        skipna=True).sort_values(ascending=False).head(n)
    result = movies[movies['movieId'].isin(
        avg_scores.index)][['movieId','title','genres']].copy()
    result = result.set_index('movieId')
    result['predicted_rating'] = avg_scores.round(2)
    return result.sort_values(
        'predicted_rating', ascending=False).reset_index()