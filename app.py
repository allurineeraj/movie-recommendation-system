import streamlit as st
import pandas as pd
from recommender import (load_data, build_content_model,
    content_recommend, build_collab_model, collab_recommend)
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("Movie Recommendation System")
st.markdown("Built with **Content-Based** and **Collaborative Filtering** on the MovieLens dataset.")

@st.cache_data
def setup():
    movies, ratings = load_data()
    cosine_sim, indices = build_content_model(movies)
    matrix, user_sim_df = build_collab_model(ratings)
    return movies, ratings, cosine_sim, indices, matrix, user_sim_df

movies, ratings, cosine_sim, indices, matrix, user_sim_df = setup()

st.sidebar.header("Dataset Info")
st.sidebar.metric("Total Movies", f"{movies.shape[0]:,}")
st.sidebar.metric("Total Users", f"{ratings['userId'].nunique():,}")
st.sidebar.metric("Total Ratings", f"{ratings.shape[0]:,}")

tab1, tab2 = st.tabs(["Content-Based Filtering",
                       "Collaborative Filtering"])

with tab1:
    st.subheader("Find movies similar to one you liked")
    st.markdown("Recommends based on **genre similarity** using TF-IDF + Cosine Similarity.")

    movie_list = sorted(movies['title'].unique())
    selected_movie = st.selectbox(
        "Choose a movie:", movie_list, index=0)
    n_recs = st.slider("Number of recommendations:", 5, 20, 10)

    if st.button("Get Recommendations", key="cb"):
        results = content_recommend(
            selected_movie, movies, cosine_sim, indices, n_recs)
        if results.empty:
            st.error("Movie not found.")
        else:
            st.success(f"Top {n_recs} movies similar to '{selected_movie}'")
            st.dataframe(results, use_container_width=True)

with tab2:
    st.subheader("Personalised recommendations for a user")
    st.markdown("Recommends based on **similar users' ratings** using User-User Cosine Similarity.")

    user_ids = sorted(matrix.index.tolist())
    selected_user = st.selectbox(
        "Choose a User ID:", user_ids, index=0)
    n_recs2 = st.slider("Number of recommendations:", 5, 20, 10,
                         key="n2")

    if st.button("Get Recommendations", key="cf"):
        results2 = collab_recommend(
            selected_user, movies, matrix, user_sim_df, n_recs2)
        if results2.empty:
            st.error("User not found.")
        else:
            st.success(f"Top {n_recs2} recommendations for User {selected_user}")
            st.dataframe(results2, use_container_width=True)

st.divider()
st.markdown("**Dataset:** MovieLens Small (GroupLens Research) · "
            "**Built by:** Alluri Neeraj Subbaraju")