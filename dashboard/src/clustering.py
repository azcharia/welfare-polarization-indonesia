"""
src/clustering.py
Modul untuk clustering models, PCA, dan evaluasi metrik.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")

from src.data import FEATURE_COLS


# ============================================================
# MODEL DEFINITIONS
# ============================================================

CLUSTER_COLORS = {
    0: "#E8505B",  # Red - Kerentanan
    1: "#2E86AB",  # Blue - Jenuh
    2: "#F18F01",  # Orange - Transisi
    3: "#2CA02C",  # Green - Menengah Atas
}


@st.cache_data
def run_kmeans_k4(_X_scaled, _df_clean):
    """K-Means K=4 — model rekomendasi utama."""
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(_X_scaled)
    return _assign_labels(_df_clean, labels)


@st.cache_data
def run_kmeans_k3(_X_scaled, _df_clean):
    """K-Means K=3 — model baseline."""
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(_X_scaled)
    return _assign_labels(_df_clean, labels)


@st.cache_data
def run_agglomerative_ward_k3(_X_scaled, _df_clean):
    """Agglomerative Ward K=3."""
    agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = agg.fit_predict(_X_scaled)
    return _assign_labels(_df_clean, labels)


@st.cache_data
def run_spectral_k3(_X_scaled, _df_clean):
    """Spectral K=3."""
    spec = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42, n_init=10)
    labels = spec.fit_predict(_X_scaled)
    return _assign_labels(_df_clean, labels)


def _assign_labels(df, labels):
    """Assign labels to dataframe with tipologi mapping."""
    df = df.copy()
    df['Klaster'] = labels
    # Auto-detect tipologi based on cluster profiles
    tipologi_map = _auto_tipologi(df)
    df['Tipologi'] = df['Klaster'].map(tipologi_map)
    df['Tipologi'] = df['Tipologi'].fillna(f"Klaster {df['Klaster']}")
    return df


def _auto_tipologi(df):
    """Auto-detect tipologi based on cluster means."""
    pdrb_col = FEATURE_COLS[0]
    p0_col = FEATURE_COLS[2]

    profil = df.groupby('Klaster')[[pdrb_col, p0_col]].mean()

    # Highest PDRB → Episentrum Ekonomi Jenuh
    # Highest Poverty → Kerentanan Struktural
    # Rest → based on ranking

    klaster_pdrb = profil[pdrb_col].idxmax()
    klaster_miskin = profil[p0_col].idxmax()

    # Sort other clusters by PDRB descending
    others = [k for k in profil.index if k not in [klaster_pdrb, klaster_miskin]]
    others_sorted = sorted(others, key=lambda k: profil.loc[k, pdrb_col], reverse=True)

    tipologi_map = {}
    tipologi_map[klaster_pdrb] = "Episentrum Ekonomi Jenuh"
    tipologi_map[klaster_miskin] = "Kerentanan Struktural"

    # Assign remaining
    labels = ["Wilayah Transisi", "Kelas Menengah Atas", "Kelas Menengah Bawah"]
    for i, k in enumerate(others_sorted):
        tipologi_map[k] = labels[i] if i < len(labels) else f"Klaster {k}"

    return tipologi_map


@st.cache_resource
def compute_pca(_X_scaled):
    """Compute PCA dan return hasilnya."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(_X_scaled)
    explained_var = pca.explained_variance_ratio_
    return X_pca, explained_var, pca


def compute_metrics(X_scaled, labels):
    """Compute Silhouette, DBI, CH untuk label tertentu."""
    unique = np.unique(labels)
    valid = unique[unique >= 0]

    if len(valid) < 2:
        return None, None, None

    mask = labels >= 0
    sil = silhouette_score(X_scaled[mask], labels[mask])
    dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
    ch = calinski_harabasz_score(X_scaled[mask], labels[mask])

    return sil, dbi, ch


@st.cache_data
def get_cluster_profiles(_df):
    """Dapatkan profil rata-rata per klaster."""
    pdrb_col = FEATURE_COLS[0]
    tpt_col = FEATURE_COLS[1]
    p0_col = FEATURE_COLS[2]

    profil = _df.groupby('Klaster').agg({
        pdrb_col: ['mean', 'std', 'count'],
        tpt_col: ['mean', 'std'],
        p0_col: ['mean', 'std'],
    }).round(2)

    profil.columns = ['PDRB_mean', 'PDRB_std', 'Jumlah',
                      'TPT_mean', 'TPT_std',
                      'P0_mean', 'P0_std']
    profil = profil.reset_index()

    # Add tipologi
    if 'Tipologi' in _df.columns:
        tipologi_map = _df.groupby('Klaster')['Tipologi'].first().to_dict()
        profil['Tipologi'] = profil['Klaster'].map(tipologi_map)

    return profil


@st.cache_data
def get_cluster_summary(_df):
    """Summary statistik per klaster (numeric, format di display layer)."""
    if 'Klaster' not in _df.columns or 'Tipologi' not in _df.columns:
        return pd.DataFrame()

    summary = _df.groupby(['Klaster', 'Tipologi']).agg(
        Jumlah_Wilayah=('Kab/Kota', 'count'),
        Rata_PDRB=(FEATURE_COLS[0], 'mean'),
        Rata_TPT=(FEATURE_COLS[1], 'mean'),
        Rata_P0=(FEATURE_COLS[2], 'mean'),
    ).round(2).reset_index()

    summary['Persentase'] = (summary['Jumlah_Wilayah'] / summary['Jumlah_Wilayah'].sum() * 100).round(1)

    return summary.sort_values('Klaster')
