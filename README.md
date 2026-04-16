# Movie Recommendation System

Live demo: [Click here](https://huggingface.co/spaces/YOUR_USERNAME/movie-recommendation-system)

A fully deployed dual recommendation engine built on the
MovieLens dataset with an interactive Streamlit web app.

## Approaches
| Method | Technique | Input |
|---|---|---|
| Content-Based | TF-IDF + Cosine Similarity | Movie genres |
| Collaborative | User-User Cosine Similarity | User ratings |

## Dataset
MovieLens Small — 9,742 movies · 610 users · 100,836 ratings
Source: grouplens.org/datasets/movielens/latest

## Key Insights
- MedInc strongest predictor in content similarity
- Content-based solves cold-start problem for new users
- Collaborative filtering improves with more rating data

## Tech Stack
Python · Scikit-learn · Pandas · Streamlit · Hugging Face Spaces

## Run locally
pip install -r requirements.txt
streamlit run app.py