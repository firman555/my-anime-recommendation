...

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
