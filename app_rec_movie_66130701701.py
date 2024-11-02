import streamlit as st
import pickle
from surprise import SVD

# Load the model and data
@st.cache_data
def load_data():
    with open('66130701701recommendation_movie_svd.pkl', 'rb') as file:
        svd_model, movie_ratings, movies = pickle.load(file)
    return svd_model, movie_ratings, movies

svd_model, movie_ratings, movies = load_data()

# Title for the Streamlit app
st.title("Movie Recommendation System")

# Input for user ID
user_id = st.number_input("Enter User ID:", min_value=1, step=1, value=1)

# Fetch movies rated by the user
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

# Get recommendations
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
top_recommendations = sorted_predictions[:10]

# Display top 10 movie recommendations
st.write(f"### Top 10 movie recommendations for User {user_id}:")
for recommendation in top_recommendations:
    movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
    st.write(f"**{movie_title}** (Estimated Rating: {recommendation.est:.2f})")


