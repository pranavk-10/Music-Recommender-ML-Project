import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# -------------------------------
# Load & Preprocess Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_millsongdata.csv")
    df = df.drop(columns=['link'], errors='ignore')  # Drop 'link' if exists
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

# -------------------------------
# Preprocessing + Vectorization
# -------------------------------
@st.cache_resource
def preprocess_and_vectorize(df):
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    df['cleared_text'] = df['text'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['cleared_text'])

    nn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)

    return df, tfidf_matrix, vectorizer, nn_model

df, tfidf_matrix, tfidf_vectorizer, nn_model = preprocess_and_vectorize(df)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_songs(song_name, model=nn_model, df=df, tfidf_matrix=tfidf_matrix, top_n=5):
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "❌ Song Not Found in Dataset", None
    idx = idx[0]

    song_vector = tfidf_matrix[idx]
    distances, indices = model.kneighbors(song_vector, n_neighbors=top_n+1)
    song_indices = indices[0][1:]  # skip the input song itself

    return df[['artist', 'song']].iloc[song_indices], idx

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🎵 Lyrics-Based Song Recommender")
st.markdown("Enter a **song name** to get 5 similar song recommendations based on lyrics.")

# User input
song_input = st.text_input("🎶 Enter a song name")

# Display Recommendations
if song_input:
    recommendations, index = recommend_songs(song_input)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.success(f"✅ Top 5 Recommendations for: *{song_input}*")
        st.dataframe(recommendations.reset_index(drop=True), use_container_width=True)

        # User feedback
        satisfaction = st.slider("How satisfied are you with the recommendations?", 1, 5, 3)
        st.write(f"⭐ You rated the recommendation as **{satisfaction}/5**. Thank you!")

# Optional: Show top artists
with st.expander("📊 Show Top 10 Most Frequent Artists"):
    top_artists = df['artist'].value_counts().head(10)
    st.bar_chart(top_artists)
