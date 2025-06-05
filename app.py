


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

st.set_page_config(page_title="🎌 Rekomendasi Anime 🎌", layout="wide")

st.markdown("<h1 style='text-align: center;'>🎌 Rekomendasi Anime 🎌</h1>", unsafe_allow_html=True)
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
    
    anime = download_and_load_csv(anime_file_id, "anime.csv")
    anime.columns = anime.columns.str.strip().str.lower()
    anime = anime[["anime_id", "name"]].dropna().drop_duplicates(subset="name")

    ratings = download_and_load_csv(rating_file_id, "rating.csv")
    ratings.columns = ratings.columns.str.strip().str.lower()
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

@st.cache_data(show_spinner=False)
def get_latest_anime(n=10):
    try:
        response = requests.get("https://api.jikan.moe/v4/seasons/now", timeout=10)
        if response.status_code == 200:
            results = []
            for anime in response.json()["data"][:n]:
                anime_id = anime["mal_id"]
                title = anime["title"]
                image = anime["images"]["jpg"].get("image_url", "")
                synopsis_en = anime.get("synopsis", "Sinopsis tidak tersedia.")
                synopsis_id = GoogleTranslator(source='auto', target='id').translate(synopsis_en)
                genres = ", ".join([g["name"] for g in anime.get("genres", [])])
                type_ = anime.get("type", "-")
                episodes = anime.get("episodes", "?")
                year = anime.get("year", "-")
                results.append({
                    "id": anime_id, "title": title, "image": image,
                    "synopsis": synopsis_id, "genres": genres,
                    "type": type_, "episodes": episodes, "year": year
                })
            return results
    except Exception as e:
        print(f"[ERROR latest] {e}")
    return []

with st.spinner("🔄 Memuat data..."):
    anime, data = load_data()
    matrix = prepare_matrix(data)
    model = train_model(matrix)
    anime_id_map = dict(zip(anime['name'], anime['anime_id']))


# ================================
# ANIME TERBARU
# ================================
st.subheader("🆕 Anime Terbaru (Season Now)")
latest = get_latest_anime()
if latest:
    col_rows = [st.columns(5), st.columns(5)]
    for i, anime_item in enumerate(latest):
        row = 0 if i < 5 else 1
        col = col_rows[row][i % 5]
        with col:
            tampilkan_gambar_anime(anime_item["image"], anime_item["title"])
            st.markdown(f"🎭 Genre: {anime_item['genres']}")
            st.markdown(f"🎮 Tipe: `{anime_item['type']}`")
            st.markdown(f"📺 Episode: `{anime_item['episodes']}`")
            st.markdown(f"🗓️ Tahun Rilis: `{anime_item['year']}`")
            with st.expander("📓 Lihat Sinopsis"):
                st.markdown(anime_item["synopsis"])
else:
    st.info("Tidak dapat memuat anime terbaru.")

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
        if len(results) >= 10:
            break

    if results:
        col_rows = [st.columns(5), st.columns(5)]
        for i, (anime_id, rating, num_votes) in enumerate(results):
            row = 0 if i < 5 else 1
            col = col_rows[row][i % 5]
            with col:
                anime_id_column = next((col for col in anime.columns if col.strip().lower() == 'anime_id'), None)
                name_column = next((col for col in anime.columns if col.strip().lower() == 'name'), None)
                if anime_id_column and name_column:
                    name_row = anime[anime[anime_id_column] == anime_id]
                    name = name_row[name_column].values[0] if not name_row.empty else "Judul Tidak Diketahui"
                else:
                    name = "Judul Tidak Diketahui"
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

st.markdown("## 🎏 Rekomendasi Berdasarkan Anime Favorit Kamu")
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



@st.cache_data(show_spinner=False)
def get_trending_anime(n=10):
    try:
        response = requests.get("https://api.jikan.moe/v4/top/anime", timeout=10)
        if response.status_code == 200:
            trending = []
            for anime in response.json()["data"][:n]:
                anime_id = anime["mal_id"]
                title = anime["title"]
                image = anime["images"]["jpg"].get("image_url", "")
                synopsis_en = anime.get("synopsis", "Sinopsis tidak tersedia.")
                synopsis_id = GoogleTranslator(source='auto', target='id').translate(synopsis_en)
                genres = ", ".join([g["name"] for g in anime.get("genres", [])])
                type_ = anime.get("type", "-")
                episodes = anime.get("episodes", "?")
                aired_from = anime.get("aired", {}).get("from", None)
                try:
                    year = pd.to_datetime(aired_from).year if aired_from else "-"
                except:
                    year = "-"
                trending.append({
                    "id": anime_id, "title": title, "image": image,
                    "synopsis": synopsis_id, "genres": genres,
                    "type": type_, "episodes": episodes, "year": year
                })
            return trending
    except Exception as e:
        print(f"[ERROR trending global] {e}")
    return []


# ================================
# ANIME TRENDING GLOBAL
# ================================
st.markdown("## 🌍 Anime Trending Global (Peringkat Teratas MyAnimeList)")

trending = get_trending_anime(10)
if trending:
    col_rows = [st.columns(5), st.columns(5)]
    for i, anime in enumerate(trending):
        row = 0 if i < 5 else 1
        col = col_rows[row][i % 5]
        with col:
            tampilkan_gambar_anime(anime["image"], anime["title"])
            st.markdown(f"🎭 Genre: {anime['genres']}")
            st.markdown(f"🎮 Tipe: `{anime['type']}`")
            st.markdown(f"📺 Episode: `{anime['episodes']}`")
            st.markdown(f"🗓️ Tahun Rilis: `{anime['year']}`")
            with st.expander("📓 Lihat Sinopsis"):
                st.markdown(anime["synopsis"])
else:
    st.info("Tidak dapat memuat anime trending global.")




st.markdown("## 🎲 Rekomendasi Acak Berkualitas (Surprise Me!)")

if st.button("🎉 Beri Saya Rekomendasi!"):
    st.subheader("🎁 Anime Rekomendasi Acak untuk Kamu:")
    
    # Filter anime dengan rating bagus
    anime_stats = data.groupby("anime_id").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    good_anime = anime_stats[(anime_stats["avg_rating"] >= 7.5) & (anime_stats["num_ratings"] > 30)]
    
    # Pilih acak
    sampled = good_anime.sample(n=5, random_state=int(time.time()))
    
    col_rows = [st.columns(5)]
    for i, row in enumerate(sampled.itertuples()):
        col = col_rows[0][i % 5]
        with col:
            name_row = anime[anime["anime_id"] == row.anime_id]
            name = name_row["name"].values[0] if not name_row.empty else "Judul Tidak Diketahui"
            image_url, synopsis, genres, type_, episodes, year = get_anime_details_cached(row.anime_id)
            
            tampilkan_gambar_anime(image_url, name)
            st.markdown(f"⭐ Rating: `{row.avg_rating:.2f}`")
            st.markdown(f"👥 Jumlah Rating: `{row.num_ratings}`")
            st.markdown(f"🎮 Tipe: `{type_}`")
            st.markdown(f"📺 Episode: `{episodes}`")
            st.markdown(f"🗓️ Tahun Rilis: `{year}`")
            st.markdown(f"🎭 Genre: {genres}")
            with st.expander("📓 Lihat Sinopsis"):
                st.markdown(synopsis)
