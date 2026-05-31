"""
app.py — Dashboard Streamlit
Polarisasi Kesejahteraan dan Kejenuhan Wilayah
Analisis Clustering dengan K-Means K=4

Struktur Modular:
  - src/data.py       : Data loading & preprocessing
  - src/clustering.py  : Models, PCA, metrics
  - src/viz.py         : Plotly visualizations
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data import (
    load_dataset, get_data_clean, get_features_scaled,
    get_provinces, filter_dataframe,
    FEATURE_COLS, FEATURE_SHORT, format_pdrb,
)
from src.clustering import (
    run_kmeans_k4, run_kmeans_k3, run_agglomerative_ward_k3,
    run_spectral_k3, compute_pca, compute_metrics,
    get_cluster_profiles, get_cluster_summary,
)
from src.viz import (
    plot_pca_scatter, plot_cluster_radar, plot_cluster_bars,
    plot_tipologi_distribution, format_dataframe_display,
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Polarisasi Kesejahteraan Indonesia",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SIDEBAR — NAVIGASI & FILTER
# ============================================================
def setup_sidebar(df_raw):
    with st.sidebar:
        st.markdown("## 🌏 **Polarisasi Kesejahteraan**")
        st.markdown("Analisis Clustering 514 Kabupaten/Kota")
        st.divider()

        # Pilihan Model / Algoritma
        st.markdown("### 🎯 Pilih Model")
        model_option = st.selectbox(
            "Algoritma Clustering",
            options=[
                "K-Means K=4 (Rekomendasi 🏆)",
                "K-Means K=3 (Baseline)",
                "Agglomerative Ward K=3",
                "Spectral K=3",
            ],
            index=0,
            help="Pilih model clustering yang ingin dieksplorasi",
        )
        st.caption("Rekomendasi: K-Means K=4 — unggul di CH & Gap Statistic")

        st.divider()

        # Filter Wilayah
        st.markdown("### 🔍 Filter Wilayah")
        provinsi_list = get_provinces(df_raw)

        prov_filter = st.multiselect(
            "Provinsi",
            options=provinsi_list,
            default=[],
            placeholder="Semua Provinsi",
        )

        # Range slider
        df_clean = get_data_clean(df_raw)
        pdrb_min, pdrb_max = float(df_clean[FEATURE_COLS[0]].min()), float(df_clean[FEATURE_COLS[0]].max())
        tpt_min, tpt_max = float(df_clean[FEATURE_COLS[1]].min()), float(df_clean[FEATURE_COLS[1]].max())
        p0_min, p0_max = float(df_clean[FEATURE_COLS[2]].min()), float(df_clean[FEATURE_COLS[2]].max())

        # Range filter langsung di sidebar dengan visual yang sangat bersih dan bebas tumpang tindih
        st.markdown("#### 💰 PDRB")
        pdrb_range = st.slider(
            "PDRB (Rupiah)",
            min_value=pdrb_min, max_value=pdrb_max,
            value=(pdrb_min, pdrb_max),
            format="%.0f",
            label_visibility="collapsed",
        )
        st.caption(f"Rentang: **{format_pdrb(pdrb_range[0])}** - **{format_pdrb(pdrb_range[1])}**")

        st.markdown("#### 📊 TPT (%)")
        tpt_range = st.slider(
            "TPT (%)",
            min_value=tpt_min, max_value=tpt_max,
            value=(tpt_min, tpt_max),
            format="%.1f",
            label_visibility="collapsed",
        )
        st.caption(f"Rentang: **{tpt_range[0]:.1f}%** - **{tpt_range[1]:.1f}%**")

        st.markdown("#### 📉 P0 - Kemiskinan (%)")
        p0_range = st.slider(
            "P0 (%)",
            min_value=p0_min, max_value=p0_max,
            value=(p0_min, p0_max),
            format="%.1f",
            label_visibility="collapsed",
        )
        st.caption(f"Rentang: **{p0_range[0]:.1f}%** - **{p0_range[1]:.1f}%**")

        st.divider()
        st.caption("📌 Data: BPS — 514 Kabupaten/Kota")
        st.caption("© 2026 — Naufal A, Rahma, Sita")

    return {
        "model": model_option,
        "prov_filter": prov_filter,
        "pdrb_range": pdrb_range,
        "tpt_range": tpt_range,
        "p0_range": p0_range,
    }


def tipologi_to_klaster(df, selected_tipologi):
    """Convert tipologi names to klaster numbers for filtering."""
    if not selected_tipologi:
        return None
    mapping = df.groupby('Klaster')['Tipologi'].first()
    return [k for k, v in mapping.items() if v in selected_tipologi]


# ============================================================
# MAIN CONTENT
# ============================================================
def main():
    st.markdown("""
        <style>
        /* Impor Google Font Outfit & Plus Jakarta Sans */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
        
        /* Terapkan Font secara Global hanya pada tag teks yang spesifik di dalam stApp agar Ikon Bawaan Streamlit Tetap Utuh */
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label, .stApp li {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        
        /* Header Utama (Outfit font) - Menggunakan Warna Tema Nativ */
        .main-header {
            font-family: 'Outfit', sans-serif !important;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            letter-spacing: -0.03em;
            color: var(--text-color) !important;
        }
        
        .sub-header {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 1.15rem;
            color: var(--text-color) !important;
            opacity: 0.85;
            font-weight: 500;
            margin-top: -0.2rem;
            letter-spacing: -0.01em;
        }
        
        /* Desain Theme-Aware Metric Cards - Super Ringkas & Anti Overflow */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color) !important;
            border-radius: 12px !important;
            padding: 0.8rem 1rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03) !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            overflow: hidden !important;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04) !important;
            border-color: var(--primary-color) !important;
        }
        
        /* Metric Labels & Values styling - Teroptimasi & Responsif */
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: var(--text-color) !important;
            opacity: 0.65 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        
        div[data-testid="stMetricValue"] > div {
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            color: var(--text-color) !important;
            letter-spacing: -0.02em !important;
            white-space: nowrap !important;
        }
        
        /* Pill-based Tab Navigation - Modern dengan Indikator Garis Bawah Premium */
        button[role="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: var(--text-color) !important;
            opacity: 0.75 !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 0.6rem 1.2rem !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            background-color: transparent !important;
            margin-right: 0.3rem !important;
        }
        
        button[role="tab"]:hover {
            color: var(--text-color) !important;
            opacity: 1 !important;
            background-color: var(--secondary-background-color) !important;
        }
        
        button[role="tab"][aria-selected="true"] {
            color: var(--primary-color) !important;
            background-color: rgba(46, 134, 171, 0.08) !important;
            border-bottom: 3px solid var(--primary-color) !important;
            opacity: 1 !important;
            font-weight: 700 !important;
        }
        
        /* Custom Divider */
        hr {
            margin: 1.5rem 0 !important;
            border-color: rgba(128, 128, 128, 0.12) !important;
            opacity: 0.7 !important;
        }
        
        /* Sidebar styling - Elegant dengan Border Tipis */
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color) !important;
            border-right: 1px solid rgba(128, 128, 128, 0.12) !important;
        }
        
        section[data-testid="stSidebar"] hr {
            border-color: rgba(128, 128, 128, 0.15) !important;
        }
        
        section[data-testid="stSidebar"] .stMarkdown h2, 
        section[data-testid="stSidebar"] .stMarkdown h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # LOAD DATA (sebelum sidebar)
    # ============================================================
    with st.spinner("Memuat data..."):
        df_raw = load_dataset()
        st.session_state["df_raw"] = df_raw
        df_clean = get_data_clean(df_raw)
        X_scaled, scaler = get_features_scaled(df_clean)

    # Header
    st.markdown('<p class="main-header">🌏 Polarisasi Kesejahteraan Indonesia</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analisis Clustering 514 Kabupaten/Kota — '
        'Deteksi Paradoks Pertumbuhan Ekonomi</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Setup sidebar & get filters (pass df_raw)
    filters = setup_sidebar(df_raw)

    # ============================================================
    # RUN CLUSTERING MODEL
    # ============================================================
    model_key = filters["model"]

    with st.spinner(f"Menjalankan {model_key}..."):
        if "K-Means K=4" in model_key:
            df_result = run_kmeans_k4(X_scaled, df_clean)
        elif "K-Means K=3" in model_key:
            df_result = run_kmeans_k3(X_scaled, df_clean)
        elif "Agglomerative" in model_key:
            df_result = run_agglomerative_ward_k3(X_scaled, df_clean)
        elif "Spectral" in model_key:
            df_result = run_spectral_k3(X_scaled, df_clean)
        else:
            df_result = run_kmeans_k4(X_scaled, df_clean)

    # Compute PCA
    X_pca, explained_var, _ = compute_pca(X_scaled)

    # Compute metrics
    sil, dbi, ch = compute_metrics(X_scaled, df_result['Klaster'].values)

    # ============================================================
    # METRIC ROW
    # ============================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Total Wilayah",
            value=f"{len(df_result)}",
            help="Jumlah kabupaten/kota yang dianalisis",
        )
    with col2:
        st.metric(
            label="Jumlah Klaster",
            value=f"{df_result['Klaster'].nunique()}",
            help="Jumlah cluster yang terbentuk",
        )
    with col3:
        st.metric(
            label="Silhouette Score",
            value=f"{sil:.4f}" if sil else "N/A",
            delta="↑ lebih baik" if sil and sil > 0.25 else None,
            help="Nilai >0.25 = baik untuk data sosial",
        )
    with col4:
        st.metric(
            label="Davies-Bouldin (DBI)",
            value=f"{dbi:.4f}" if dbi else "N/A",
            delta="↓ lebih baik" if dbi and dbi < 1.0 else None,
            help="Nilai <1.0 = baik",
        )
    with col5:
        st.metric(
            label="Calinski-Harabasz",
            value=f"{ch:.1f}" if ch else "N/A",
            help="↑ tinggi = cluster lebih kompak & terpisah",
        )

    st.divider()

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Scatter Plot PCA",
        "🗺️ Profil Klaster",
        "📋 Data Wilayah",
        "📈 Perbandingan Algoritma",
        "📝 Ringkasan & Insight",
    ])

    # ============================================================
    # TAB 1: PCA SCATTER PLOT
    # ============================================================
    with tab1:
        st.markdown("### Visualisasi Klaster — PCA 2 Dimensi")
        st.caption("Hover pada titik untuk melihat detail wilayah. Proyeksi PCA merangkum "
                   f"{explained_var[0]*100+explained_var[1]*100:.1f}% varians data dari 3 fitur.")

        # Filter data for display
        df_filtered = filter_dataframe(
            df_result,
            filters["prov_filter"],
            filters["pdrb_range"],
            filters["tpt_range"],
            filters["p0_range"],
        )

        # Merge with PCA
        df_pca = df_filtered.copy()
        df_pca['PC1'] = X_pca[df_filtered.index, 0]
        df_pca['PC2'] = X_pca[df_filtered.index, 1]

        # Short labels for hover
        df_pca['PDRB'] = df_pca[FEATURE_COLS[0]].apply(lambda x: f"Rp{x/1e9:.1f}M" if x >= 1e9 else f"Rp{x:,.0f}")
        df_pca['TPT'] = df_pca[FEATURE_COLS[1]]
        df_pca['P0'] = df_pca[FEATURE_COLS[2]]

        col_left, col_right = st.columns([3, 1])
        with col_left:
            fig = plot_pca_scatter(df_pca, explained_var)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### Distribusi")
            fig_dist = plot_tipologi_distribution(df_filtered)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("#### Varians PCA")
            st.markdown(f"""
            - **PC1**: {explained_var[0]*100:.1f}%
            - **PC2**: {explained_var[1]*100:.1f}%
            - **Total**: {sum(explained_var)*100:.1f}%
            """)

            st.markdown("#### Filter Aktif")
            if filters["prov_filter"]:
                st.info(f"Provinsi: {', '.join(filters['prov_filter'])}")
            else:
                st.info("Semua Provinsi")

    # ============================================================
    # TAB 2: CLUSTER PROFILES
    # ============================================================
    with tab2:
        st.markdown("### Profil & Karakteristik Klaster")

        profil = get_cluster_profiles(df_result)
        summary = get_cluster_summary(df_result)

        # Summary table
        st.markdown("#### Ringkasan per Klaster")
        if not summary.empty:
            # Styled dataframe
            st.dataframe(
                summary,
                column_config={
                    "Klaster": st.column_config.NumberColumn("Klaster", format="%d"),
                    "Tipologi": "Tipologi",
                    "Jumlah_Wilayah": st.column_config.NumberColumn("Jumlah Wilayah", format="%d"),
                    "Persentase": st.column_config.NumberColumn("Persentase", format="%.1f%%"),
                    "Rata_PDRB": st.column_config.NumberColumn("Rata-rata PDRB (Rp)", format="Rp%.0f"),
                    "Rata_TPT": st.column_config.NumberColumn("Rata-rata TPT (%)", format="%.2f"),
                    "Rata_P0": st.column_config.NumberColumn("Rata-rata P0 (%)", format="%.2f"),
                },
                hide_index=True,
                use_container_width=True,
            )

        col_left, col_right = st.columns(2)
        with col_left:
            # Radar chart
            fig_radar = plot_cluster_radar(profil)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_right:
            # Bar chart
            fig_bar = plot_cluster_bars(profil)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Contoh wilayah per klaster
        st.markdown("#### Contoh Wilayah per Klaster")
        tipologi_cols = st.columns(4) if df_result['Klaster'].nunique() >= 4 else st.columns(df_result['Klaster'].nunique())
        for i, (col, tipologi_name) in enumerate(zip(tipologi_cols, sorted(df_result['Tipologi'].unique()))):
            subset = df_result[df_result['Tipologi'] == tipologi_name]
            with col:
                st.markdown(f"**{tipologi_name}** ({len(subset)} wilayah)")
                for _, row in subset.head(5).iterrows():
                    st.markdown(f"- {row['Kab/Kota']}")

    # ============================================================
    # TAB 3: DATA TABLE
    # ============================================================
    with tab3:
        st.markdown("### Data Eksplorasi Wilayah")
        st.caption("Cari, filter, dan urutkan data 514 kabupaten/kota berdasarkan provinsi, "
                   "klaster, atau indikator ekonomi.")

        # Search
        search = st.text_input("🔍 Cari kabupaten/kota", placeholder="Ketik nama wilayah...")

        # Filter by cluster
        cluster_options = sorted(df_result['Tipologi'].unique())
        cluster_filter = st.multiselect(
            "Filter Tipologi",
            options=cluster_options,
            default=cluster_options,
            placeholder="Semua tipologi",
        )

        # Apply filters
        df_display = filter_dataframe(
            df_result,
            filters["prov_filter"],
            filters["pdrb_range"],
            filters["tpt_range"],
            filters["p0_range"],
            cluster_filter=tipologi_to_klaster(df_result, cluster_filter),
        )

        # Search filter
        if search:
            mask = df_display['Kab/Kota'].str.contains(search, case=False, na=False)
            df_display = df_display[mask]

        st.markdown(f"Menampilkan **{len(df_display)}** dari **{len(df_result)}** wilayah")

        # Show data
        df_show = format_dataframe_display(df_display)
        st.dataframe(
            df_show,
            column_config={
                "Provinsi": st.column_config.TextColumn("Provinsi"),
                "Kab/Kota": st.column_config.TextColumn("Kab/Kota"),
                "Tipologi": st.column_config.TextColumn("Tipologi"),
                "Klaster": st.column_config.NumberColumn("Klaster", format="%d"),
                "PDRB (Rp)": st.column_config.NumberColumn("PDRB (Rp)", format="Rp%.0f"),
                "TPT (%)": st.column_config.NumberColumn("TPT (%)", format="%.2f"),
                "P0 (%)": st.column_config.NumberColumn("P0 (%)", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=500,
        )

        # Download button
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="hasil_clustering_polarisasi.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ============================================================
    # TAB 4: MODEL COMPARISON
    # ============================================================
    with tab4:
        st.markdown("### Perbandingan Semua Algoritma")
        st.caption("Perbandingan 12 konfigurasi model dengan 5 metrik evaluasi clustering.")

        # Load comparison data
        try:
            APP_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_DIR = os.path.dirname(APP_DIR)
            compare_path = os.path.join(PROJECT_DIR, "eksperimen", "28_evaluasi_metrik_lengkap", "perbandingan_5_metrik.csv")
            if not os.path.exists(compare_path):
                compare_path = os.path.join("eksperimen", "28_evaluasi_metrik_lengkap", "perbandingan_5_metrik.csv")
            df_compare = pd.read_csv(compare_path)
            df_compare = df_compare.rename(columns={
                "Calinski_Harabasz": "CH",
                "Dunn_Index": "Dunn",
                "Gap_Statistic": "Gap",
            })
            df_compare = df_compare[["Model", "Klaster", "Silhouette", "DBI", "CH", "Dunn", "Gap"]]

            # Highlight best model
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(
                    df_compare,
                    column_config={
                        "Model": st.column_config.TextColumn("Model"),
                        "Klaster": st.column_config.NumberColumn("K", format="%d"),
                        "Silhouette": st.column_config.NumberColumn("Silhouette↑", format="%.4f"),
                        "DBI": st.column_config.NumberColumn("DBI↓", format="%.4f"),
                        "CH": st.column_config.NumberColumn("CH↑", format="%.1f"),
                        "Dunn": st.column_config.NumberColumn("Dunn↑", format="%.4f"),
                        "Gap": st.column_config.NumberColumn("Gap↑", format="%.4f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=450,
                )

            with col2:
                st.markdown("#### 🏆 Model Unggulan")
                st.markdown("""
                **K-Means K=4**
                - CH: 386.5 🥇
                - Gap: 1.589 🥇
                - Silhouette: 0.432 ✅
                - DBI: 0.752 ✅

                **Agglomerative Ward K=3**
                - Silhouette: 0.472 🥇 (3-K)
                - Interpretasi hierarkis

                **Catatan:**
                Dunn Index rendah karena
                outlier alami (Jakarta, Papua)
                """)

            # Bar chart comparison
            st.divider()
            from src.viz import plot_metrics_comparison
            fig_comp = plot_metrics_comparison(df_compare)
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)

        except FileNotFoundError:
            st.warning("⚠️ File perbandingan metrik belum tersedia. "
                       "Jalankan `eksperimen/run_metrik_lengkap.py` terlebih dahulu.")

    # ============================================================
    # TAB 5: SUMMARY & INSIGHTS
    # ============================================================
    with tab5:
        st.markdown("### 📝 Ringkasan & Insight")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### 🎯 Temuan Utama")
            st.markdown("""
            **1. Paradoks Pertumbuhan Ekonomi Terkonfirmasi ✅**
            - Jakarta, Surabaya, Bandung, Bekasi, Makassar masuk klaster
              "Episentrum Ekonomi Jenuh" — PDRB tinggi tapi pengangguran tinggi.
            - Ini mengonfirmasi adanya kejenuhan industrialisasi di kota besar.

            **2. Ketimpangan Regional Masih Sangat Tajam 📊**
            - Wilayah Indonesia Timur (Papua, NTT, Maluku) mendominasi
              klaster "Kerentanan Struktural" — kemiskinan di atas 25%.
            - 80%+ wilayah masuk klaster menengah — mayoritas homogen.

            **3. K-Means K=4 Terbukti Unggul 🏆**
            - Calinski-Harabasz: 386.5 (tertinggi)
            - Gap Statistic: 1.589 (tertinggi)
            - Distribusi klaster lebih seimbang dari model lain.
            """)

        with col_right:
            # Quick stats
            st.markdown("#### 📊 Statistik Cepat")
            tipologi_counts = df_result['Tipologi'].value_counts()

            for tipologi, count in tipologi_counts.items():
                pct = count / len(df_result) * 100
                color = "#E8505B" if "Kerentanan" in tipologi else \
                        "#2E86AB" if "Jenuh" in tipologi else \
                        "#F18F01" if "Transisi" in tipologi else "#2CA02C"
                st.markdown(f"""
                <div style="
                    background-color: var(--secondary-background-color);
                    color: var(--text-color) !important;
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    padding: 0.8rem 1.2rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
                    border: 1px solid rgba(128, 128, 128, 0.12);
                    word-wrap: break-word;
                ">
                    <strong style="color: {color}; font-size: 0.95rem; font-weight: 700; font-family: 'Outfit', sans-serif;">{tipologi}</strong><br>
                    <span style="font-size: 1.6rem; font-weight: 800; font-family: 'Outfit', sans-serif; color: var(--text-color) !important;">{count}</span>
                    <span style="color: var(--text-color) !important; opacity: 0.7; font-size: 0.9rem;"> wilayah ({pct:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.markdown("#### 📈 Metrik Model Saat Ini")
            st.markdown(f"""
            - **Model**: {model_key.split(" (")[0]}
            - **Silhouette**: {sil:.4f}
            - **DBI**: {dbi:.4f}
            - **CH**: {ch:.1f}
            - **Total Varians PCA**: {sum(explained_var)*100:.1f}%
            """)

        # Kesimpulan
        st.divider()
        st.markdown("#### 💡 Kesimpulan untuk HKI")
        st.success(
            "Model **K-Means K=4** dengan fitur PDRB, TPT, dan P0 adalah model final yang "
            "direkomendasikan untuk pengajuan HKI Program Komputer. "
            "Model ini unggul di Calinski-Harabasz (386.5) dan Gap Statistic (1.589), "
            "serta menghasilkan 4 tipologi wilayah yang interpretable: "
            "**Episentrum Ekonomi Jenuh**, **Kelas Menengah Atas**, "
            "**Wilayah Transisi**, dan **Kerentanan Struktural**."
        )


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
