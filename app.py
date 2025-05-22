
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
from rapidfuzz import process

st.set_page_config(page_title="🍜 Sistem Rekomendasi Anime", layout="wide")
st.markdown("<h1 style='text-align: center;'>🍜 Sistem Rekomendasi Anime</h1>", unsafe_allow_html=True)
st.caption("Powered by K-Nearest Neighbors, Jikan API & Google Drive")

AVAILABLE_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Slice of Life", "Supernatural", "Sports", "Thriller"
]

def tampilkan_gambar_anime(image_url, caption):
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='{image_url if image_url else "https://via.placeholder.com/200x300?text=No+Image"}'
                 style='height: 300px; object-fit: cover; border-radius: 10px;'>
            <p style='margin-top: 6px; font-weight: bold;'>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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

@st.cache_data
def prepare_matrix(data, num_users=5500, num_anime=5000):
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
            aired_from = data.get("aired", {}).get("from", None)
            year = "-"
            aired_from = data.get("aired", {}).get("from", None)
            year = "-"
            if aired_from:
                try:
                    year = pd.to_datetime(aired_from).year
                except:
                    pass

            return image, synopsis_id, genres, type_, episodes, year
    except Exception as e:
        print(f"[ERROR] ID {anime_id}: {e}")
    return "", "Sinopsis tidak tersedia.", "-", "-", "?", "-"

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

with st.spinner("🔄 Memuat data..."):
    anime, data = load_data()
    matrix = prepare_matrix(data)
    model = train_model(matrix)
    anime_id_map = dict(zip(anime['name'], anime['anime_id']))

# PENCARIAN MANUAL

st.markdown("## 🔍 Cari Anime Manual")

search_query = st.text_input("Ketik nama anime:")

if search_query:
    matches = process.extract(search_query, anime["name"].tolist(), limit=10, score_cutoff=60)
    matching_titles = [match[0] for match in matches]

    if matching_titles:
        selected_title = st.selectbox("Pilih anime yang dimaksud:", matching_titles)
        anime_id = anime_id_map.get(selected_title)
        if anime_id:
            image_url, synopsis, genres, type_, episodes, year = get_anime_details_cached(anime_id)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <img src='{image_url if image_url else "https://via.placeholder.com/200x300?text=No+Image"}'
                         style='height: 300px; object-fit: cover; border-radius: 10px;'>
                    <p style='margin-top: 10px; font-size: 18px;'><strong>Tipe:</strong> {type_}</p>
                    <p style='font-size: 18px;'><strong>📺 Total Episode:</strong> {episodes}</p>
                    <p style='font-size: 18px;'><strong>🗓️ Tahun Rilis:</strong> {year}</p>
                    <p style='font-size: 18px;'><strong>🎭 Genre:</strong> {genres}</p>
                </div>
            """, unsafe_allow_html=True)
            with st.expander("📓 Sinopsis"):
                st.markdown(synopsis)
    else:
        st.warning("Tidak ditemukan anime yang cocok. Coba ketik sebagian nama atau periksa ejaan.")

# LEADERBOARD TOP 5

st.subheader("🏆 Top 5 Anime Berdasarkan Rating")
top5_df = get_top_5_anime(data)
cols = st.columns(5)

for i, row in enumerate(top5_df.itertuples()):
    with cols[i]:
        anime_id = anime_id_map.get(row.name)
        image_url, _, _, type_, episodes, year = get_anime_details_cached(anime_id) if anime_id else ("", "", "-", "-", "?", "-")
        tampilkan_gambar_anime(image_url, row.name)
        st.markdown(f"⭐ **Rating:** `{row.avg_rating:.2f}`")
        st.markdown(f"👥 **Jumlah Rating:** `{row.num_ratings}`")
        st.markdown(f"🎮 **Tipe:** `{type_}`")
        st.markdown(f"📺 **Total Episode:** `{episodes}`")
        st.markdown(f"🗓️ **Tahun Rilis:** `{year}`")
        

# REKOMENDASI BERDASARKAN GENRE

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
        if len(results) >= 10:
            break

    if results:
        col_rows = [st.columns(5), st.columns(5)]
        for i, (anime_id, rating, num_votes) in enumerate(results):
            row = 0 if i < 5 else 1
            col = col_rows[row][i % 5]
            with col:
                name = anime[anime['anime_id'] == anime_id]['name'].values[0]
                image_url, synopsis, _, type_, episodes, year = get_anime_details_cached(anime_id)
                tampilkan_gambar_anime(image_url, name)
                st.markdown(f"⭐ Rating: `{rating:.2f}`")
                st.markdown(f"👥 Jumlah Rating: `{num_votes}`")
                st.markdown(f"🎮 Tipe: `{type_}`")
                st.markdown(f"📺 Total Episode: `{episodes}`")
                st.markdown(f"🗓️ Tahun Rilis: `{year}`")
                with st.expander("📓 Lihat Sinopsis"):
                    st.markdown(synopsis)
    else:
        st.info("Tidak ada anime ditemukan untuk genre ini.")

st.markdown("## 🎮 Rekomendasi Berdasarkan Anime Favorit Kamu")
anime_list = list(matrix.index)
selected_anime = st.selectbox("Pilih anime yang kamu suka:", anime_list)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Tampilkan Rekomendasi"):
    st.session_state.history.append(selected_anime)
    rekomendasi = get_recommendations(selected_anime, matrix, model, n=10)

    st.subheader(f"✨ Rekomendasi berdasarkan: {selected_anime}")
    col_rows = [st.columns(5), st.columns(5)]
    for i, (rec_title, similarity) in enumerate(rekomendasi):
        row = 0 if i < 5 else 1
        col = col_rows[row][i % 5]
        with col:
            anime_id = anime_id_map.get(rec_title)
            image_url, synopsis, genres, type_, episodes, year = get_anime_details_cached(anime_id) if anime_id else ("", "", "-", "-", "?", "-")
            tampilkan_gambar_anime(image_url, rec_title)
            st.markdown(f"*Genre:* {genres}")
            st.markdown(f"🎮 Tipe: `{type_}`")
            st.markdown(f"📺 Total Episode: `{episodes}`")
            st.markdown(f"🗓️ Tahun Rilis: `{year}`")
            st.markdown(f"🔗 Kemiripan: `{similarity:.2f}`")
            with st.expander("📓 Lihat Sinopsis"):
                st.markdown(synopsis)

# ================================
# RIWAYAT
# ================================
if st.session_state.history:
    st.markdown("### 🕒 Riwayat Anime yang Kamu Pilih:")
    history = st.session_state.history[-5:]
    cols = st.columns(5)
    for i, title in enumerate(reversed(history)):
        col = cols[i % 5]
        with col:
            anime_id = anime_id_map.get(title)
            image_url, _, _, type_, episodes, year = get_anime_details_cached(anime_id) if anime_id else ("", "", "-", "-", "?", "-")
            tampilkan_gambar_anime(image_url, title)
            st.markdown(f"🎮 Tipe: `{type_}`")
            st.markdown(f"📺 Total Episode: `{episodes}`")
            st.markdown(f"🗓️ Tahun Rilis: `{year}`")

    if st.button("🧹 Hapus Riwayat"):
        st.session_state.history = []
