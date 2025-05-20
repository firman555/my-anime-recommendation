
import streamlit as st
import pandas as pd
import os
import gdown
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import requests
from deep_translator import GoogleTranslator
import re
import time

# ================================
# KONFIGURASI
# ================================
st.set_page_config(page_title="🍜 Sistem Rekomendasi Anime", layout="wide")
st.markdown("<h1 style='text-align: center;'>🍜 Sistem Rekomendasi Anime</h1>", unsafe_allow_html=True)
st.caption("Powered by K-Nearest Neighbors, Jikan API & Google Drive")

AVAILABLE_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Slice of Life", "Supernatural", "Sports", "Thriller"
]

# ================================
# DOWNLOAD DATA
# ================================
@st.cache_data
def download_and_load_csv(file_id, filename):
    output = f"/tmp/{filename}"
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    return pd.read_csv(output)

@st.cache_data
def load_data():
    anime_file_id = "1rKuccpP1bsiRxozgHZAaruTeDUidRwcz"
    rating_file_id = "1bSK2RJN23du0LR1K5HdCGsp8bWckVWQn"
    anime = download_and_load_csv(anime_file_id, "anime.csv")[["anime_id", "name"]].dropna().drop_duplicates(subset="name")
    ratings = download_and_load_csv(rating_file_id, "rating.csv")
    ratings = ratings[ratings["rating"] > 0]
    data = ratings.merge(anime, on="anime_id")
    return anime, data

# ================================
# MATRIX & MODEL
# ================================
@st.cache_data
def prepare_matrix(data, num_users=1500, num_anime=1200):
    top_users = data['user_id'].value_counts().head(num_users).index
    top_anime = data['name'].value_counts().head(num_anime).index
    filtered = data[data['user_id'].isin(top_users) & data['name'].isin(top_anime)]
    matrix = filtered.pivot_table(index='name', columns='user_id', values='rating').fillna(0)
    return matrix.astype('float32')

@st.cache_resource
def train_model(matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(csr_matrix(matrix.values))
    return model

def get_recommendations(title, matrix, model, n=5):
    if title not in matrix.index:
        return []
    idx = matrix.index.get_loc(title)
    dists, idxs = model.kneighbors(matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=n+1)
    return [(matrix.index[i], 1 - dists.flatten()[j]) for j, i in enumerate(idxs.flatten()[1:])]

# ================================
# JIKAN API
# ================================
@st.cache_data(show_spinner=False)
def get_anime_details_cached(anime_id):
    try:
        time.sleep(0.3)
        response = requests.get(f"https://api.jikan.moe/v4/anime/{anime_id}", timeout=10)
        if response.status_code == 200 and response.json()["data"]:
            data = response.json()["data"]
            image = data["images"]["jpg"].get("image_url", "")
            synopsis_en = data.get("synopsis", "Sinopsis tidak tersedia.")
            genres = ", ".join([g["name"] for g in data.get("genres", [])])
            synopsis_id = GoogleTranslator(source='auto', target='id').translate(synopsis_en)
            type_ = data.get("type", "-")
            episodes = data.get("episodes", "?")
            return image, synopsis_id, genres, type_, episodes
    except Exception as e:
        print(f"[ERROR] ID {anime_id}: {e}")
    return "", "Sinopsis tidak tersedia.", "-", "-", "?"

@st.cache_data(show_spinner=False)
def get_genres_by_id(anime_id):
    try:
        time.sleep(0.3)
        response = requests.get(f"https://api.jikan.moe/v4/anime/{anime_id}", timeout=10)
        if response.status_code == 200 and response.json()["data"]:
            return [g["name"] for g in response.json()["data"].get("genres", [])]
    except Exception as e:
        print(f"[ERROR genre] ID {anime_id}: {e}")
    return []

@st.cache_data
def get_top_5_anime(data):
    grouped = data.groupby("name").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    top_anime = grouped[grouped["num_ratings"] > 10].sort_values(by="avg_rating", ascending=False).head(5)
    return top_anime

# ================================
# LOAD DATA & MODEL
# ================================
with st.spinner("🔄 Memuat data..."):
    anime, data = load_data()
    matrix = prepare_matrix(data)
    model = train_model(matrix)
    anime_id_map = dict(zip(anime['name'], anime['anime_id']))

# ================================
# LEADERBOARD
# ================================
st.subheader("🏆 Top 5 Anime Berdasarkan Rating")
top5_df = get_top_5_anime(data)
cols = st.columns(5)

for i, row in enumerate(top5_df.itertuples()):
    with cols[i]:
        anime_id = anime_id_map.get(row.name)
        image_url, _, _, type_, episodes = get_anime_details_cached(anime_id) if anime_id else ("", "", "-", "-", "?")
        st.image(image_url if image_url else "https://via.placeholder.com/200x300?text=No+Image", caption=row.name, use_container_width=True)
        st.markdown(f"⭐ **Rating:** `{row.avg_rating:.2f}`")
        st.markdown(f"👥 **Jumlah Rating:** `{row.num_ratings}`")
        st.markdown(f"🎮 **Tipe:** `{type_}`")
        st.markdown(f"📺 **Total Episode:** `{episodes}`")

# ================================
# REKOMENDASI BERDASARKAN GENRE
# ================================
st.markdown("## 🎬 Rekomendasi Berdasarkan Genre")
selected_genre = st.selectbox("Pilih genre favoritmu:", AVAILABLE_GENRES)

if st.button("🌟 Tampilkan Anime Genre Ini"):
    st.subheader(f"📚 Rekomendasi Anime dengan Genre: {selected_genre}")
    anime_ratings = data.groupby("anime_id").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    top_candidates = anime_ratings[anime_ratings["num_ratings"] > 10].sort_values(by="avg_rating", ascending=False)

    results = []
    for row in top_candidates.itertuples():
        genres = get_genres_by_id(row.anime_id)
        if selected_genre in genres:
            results.append((row.anime_id, row.avg_rating, row.num_ratings))
        if len(results) >= 5:
            break

    if results:
        cols = st.columns(len(results))
        for i, (anime_id, rating, num_votes) in enumerate(results):
            with cols[i]:
                name = anime[anime['anime_id'] == anime_id]['name'].values[0]
                image_url, _, _, type_, episodes = get_anime_details_cached(anime_id)
                st.image(image_url if image_url else "https://via.placeholder.com/200x300?text=No+Image", caption=name, use_container_width=True)
                st.markdown(f"⭐ Rating: `{rating:.2f}`")
                st.markdown(f"👥 Jumlah Rating: `{num_votes}`")
                st.markdown(f"🎮 Tipe: `{type_}`")
                st.markdown(f"📺 Total Episode: `{episodes}`")
    else:
        st.info("Tidak ada anime ditemukan untuk genre ini.")

# ================================
# FITUR REKOMENDASI BERDASARKAN PILIHAN
# ================================
st.markdown("## 🎮 Pilih Anime Favorit Kamu")
anime_list = list(matrix.index)
selected_anime = st.selectbox("Pilih anime yang kamu suka:", anime_list)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Tampilkan Rekomendasi"):
    st.session_state.history.append(selected_anime)
    rekomendasi = get_recommendations(selected_anime, matrix, model, n=5)

    st.subheader(f"✨ Rekomendasi berdasarkan: {selected_anime}")
    cols = st.columns(5)
    for i, (rec_title, similarity) in enumerate(rekomendasi):
        with cols[i % 5]:
            anime_id = anime_id_map.get(rec_title)
            image_url, synopsis, genres, type_, episodes = get_anime_details_cached(anime_id) if anime_id else ("", "", "-", "-", "?")
            st.image(image_url if image_url else "https://via.placeholder.com/200x300?text=No+Image", caption=rec_title, use_container_width=True)
            st.markdown(f"*Genre:* {genres}")
            st.markdown(f"🎮 Tipe: `{type_}`")
            st.markdown(f"📺 Total Episode: `{episodes}`")
            st.markdown(f"🔗 Kemiripan: `{similarity:.2f}`")
            with st.expander("📓 Lihat Sinopsis"):
                st.markdown(synopsis)

# ================================
# RIWAYAT
# ================================
if st.session_state.history:
    st.markdown("### 🕒 Riwayat Anime yang Kamu Pilih:")
    history = st.session_state.history[-5:]
    cols = st.columns(len(history))
    for i, title in enumerate(reversed(history)):
        with cols[i]:
            anime_id = anime_id_map.get(title)
            image_url, _, _, type_, episodes = get_anime_details_cached(anime_id) if anime_id else ("", "", "-", "-", "?")
            st.image(image_url if image_url else "https://via.placeholder.com/200x300?text=No+Image", caption=title, use_container_width=True)
            st.markdown(f"🎮 Tipe: `{type_}`")
            st.markdown(f"📺 Total Episode: `{episodes}`")
