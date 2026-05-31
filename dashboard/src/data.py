"""
src/data.py
Modul untuk loading, preprocessing, dan caching dataset.
Semua fungsi menggunakan @st.cache_data untuk performa optimal.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ============================================================
# KONSTANTA
# ============================================================
# Resolve path dynamically relative to this script's location
SRC_DIR = os.path.dirname(os.path.abspath(__file__))      # dashboard/src
DASHBOARD_DIR = os.path.dirname(SRC_DIR)                  # dashboard
PROJECT_DIR = os.path.dirname(DASHBOARD_DIR)              # root HKI
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")

FEATURE_COLS = [
    'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)',
    'Tingkat Pengangguran Terbuka',
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)',
]

FEATURE_SHORT = {
    'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)': 'PDRB (Rp)',
    'Tingkat Pengangguran Terbuka': 'TPT (%)',
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)': 'P0 (%)',
}




@st.cache_data
def load_dataset():
    """Load dataset utama dan hasil clustering ( jika ada )."""
    df = pd.read_csv(DATASET_PATH)
    return df


@st.cache_data
def get_provinces(_df):
    """Dapatkan daftar provinsi unik."""
    provinsi = sorted(_df[_df['Provinsi'].notna()]['Provinsi'].unique())
    return provinsi


@st.cache_data
def get_data_clean(_df):
    """Data bersih dengan 3 fitur utama, tanpa missing values."""
    df = _df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    return df


@st.cache_data
def get_features_scaled(_df):
    """Return X_scaled (StandardScaler) dan scaler object."""
    X_raw = _df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler


@st.cache_data
def get_feature_stats(_df):
    """Statistik deskriptif fitur utama."""
    stats = _df[FEATURE_COLS].describe().round(2)
    stats = stats.rename(columns=FEATURE_SHORT)
    return stats


def format_pdrb(value):
    """Format PDRB ke string yang rapi (dalam miliar/triliun)."""
    if value >= 1e12:
        return f"Rp{value/1e12:.2f} T"
    elif value >= 1e9:
        return f"Rp{value/1e9:.2f} M"
    elif value >= 1e6:
        return f"Rp{value/1e6:.2f} Jt"
    else:
        return f"Rp{value:,.0f}"


def filter_dataframe(df, provinsi_filter, pdrb_range, tpt_range, p0_range, cluster_filter=None):
    """Filter dataframe berdasarkan parameter yang dipilih user."""
    mask = pd.Series(True, index=df.index)

    if provinsi_filter and len(provinsi_filter) > 0:
        mask &= df['Provinsi'].isin(provinsi_filter)

    pdrb_col = FEATURE_COLS[0]
    tpt_col = FEATURE_COLS[1]
    p0_col = FEATURE_COLS[2]

    if pdrb_range:
        mask &= (df[pdrb_col] >= pdrb_range[0]) & (df[pdrb_col] <= pdrb_range[1])
    if tpt_range:
        mask &= (df[tpt_col] >= tpt_range[0]) & (df[tpt_col] <= tpt_range[1])
    if p0_range:
        mask &= (df[p0_col] >= p0_range[0]) & (df[p0_col] <= p0_range[1])

    if cluster_filter is not None and 'Klaster' in df.columns:
        if isinstance(cluster_filter, list) and len(cluster_filter) > 0:
            mask &= df['Klaster'].isin(cluster_filter)

    return df[mask].copy()
