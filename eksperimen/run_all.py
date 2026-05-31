# =============================================================================
# PROGRAM EKSPERIMEN: POLARISASI KESEJAHTERAAN
# Menguji berbagai kombinasi fitur dan nilai K untuk clustering K-Means++
# =============================================================================
# Tujuan:
#   Menemukan konfigurasi fitur dan jumlah klaster (K) optimal untuk
#   mengelompokkan 514 kabupaten/kota di Indonesia berdasarkan indikator
#   kesejahteraan ekonomi dan sosial.
#
# Penulis  : Naufal A, Rahma, Sita
# Tahun    : 2026
# =============================================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (tidak perlu popup)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")
EXPERIMEN_DIR = os.path.dirname(os.path.abspath(__file__))

# Semua fitur numerik yang tersedia (kecuali Provinsi, Kab/Kota, Klasifikasi Kemiskinan)
ALL_FEATURES = {
    "PDRB": "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
    "TPT": "Tingkat Pengangguran Terbuka",
    "P0": "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
    "IPM": "Indeks Pembangunan Manusia",
    "Sanitasi": "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "AirMinum": "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    "Sekolah": "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran": "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    "UHH": "Umur Harapan Hidup (Tahun)",
    "TPAK": "Tingkat Partisipasi Angkatan Kerja",
}

# Fitur baseline (yang digunakan di script awal)
BASELINE_FEATURES = ["PDRB", "TPT", "P0"]

def get_feature_cols(names):
    """Mengembalikan list nama kolom asli dari short names."""
    return [ALL_FEATURES[n] for n in names]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINISI EKSPERIMEN
# ─────────────────────────────────────────────────────────────────────────────
# Setiap eksperimen punya:
#   - nama: untuk folder
#   - fitur: list short names
#   - K_values: list nilai K yang akan dicoba (None = pakai K optimal dari evaluasi)
#   - description: penjelasan

EXPERIMENTS = [
    {
        "nama": "01_baseline",
        "fitur": BASELINE_FEATURES,
        "K_values": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "K_fixed": 3,
        "description": "BASELINE: PDRB + TPT + P0 (reproduksi model awal, cari K optimal)",
    },
    {
        "nama": "02_k4",
        "fitur": BASELINE_FEATURES,
        "K_values": None,
        "K_fixed": 4,
        "description": "K=4: PDRB + TPT + P0 dengan 4 klaster",
    },
    {
        "nama": "03_k5",
        "fitur": BASELINE_FEATURES,
        "K_values": None,
        "K_fixed": 5,
        "description": "K=5: PDRB + TPT + P0 dengan 5 klaster",
    },
    {
        "nama": "04_pdrb_only",
        "fitur": ["PDRB"],
        "K_values": None,
        "K_fixed": 3,
        "description": "1 FITUR: PDRB saja — apakah PDRB sendiri bisa membentuk klaster bermakna?",
    },
    {
        "nama": "05_tpt_only",
        "fitur": ["TPT"],
        "K_values": None,
        "K_fixed": 3,
        "description": "1 FITUR: TPT (Pengangguran) saja",
    },
    {
        "nama": "06_p0_only",
        "fitur": ["P0"],
        "K_values": None,
        "K_fixed": 3,
        "description": "1 FITUR: P0 (Kemiskinan) saja",
    },
    {
        "nama": "07_pdrb_tpt",
        "fitur": ["PDRB", "TPT"],
        "K_values": None,
        "K_fixed": 3,
        "description": "2 FITUR: PDRB + TPT (tanpa kemiskinan)",
    },
    {
        "nama": "08_pdrb_p0",
        "fitur": ["PDRB", "P0"],
        "K_values": None,
        "K_fixed": 3,
        "description": "2 FITUR: PDRB + P0 (tanpa pengangguran)",
    },
    {
        "nama": "09_tpt_p0",
        "fitur": ["TPT", "P0"],
        "K_values": None,
        "K_fixed": 3,
        "description": "2 FITUR: TPT + P0 (tanpa PDRB) — fokus pada kerentanan sosial",
    },
    {
        "nama": "10_tambah_ipm",
        "fitur": BASELINE_FEATURES + ["IPM"],
        "K_values": None,
        "K_fixed": 3,
        "description": "TAMBAH IPM: PDRB + TPT + P0 + Indeks Pembangunan Manusia",
    },
    {
        "nama": "11_tambah_sanitasi",
        "fitur": BASELINE_FEATURES + ["Sanitasi"],
        "K_values": None,
        "K_fixed": 3,
        "description": "TAMBAH SANITASI: PDRB + TPT + P0 + Akses Sanitasi Layak",
    },
    {
        "nama": "12_tambah_semua",
        "fitur": BASELINE_FEATURES + ["IPM", "Sanitasi", "AirMinum", "Sekolah"],
        "K_values": None,
        "K_fixed": 3,
        "description": "TAMBAH SEMUA: PDRB + TPT + P0 + IPM + Sanitasi + Air Minum + Sekolah",
    },
    {
        "nama": "13_semua_fitur",
        "fitur": list(ALL_FEATURES.keys()),
        "K_values": None,
        "K_fixed": 3,
        "description": "SEMUA FITUR: PDRB + TPT + P0 + IPM + Sanitasi + Air Minum + Sekolah + Pengeluaran + UHH + TPAK",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI-FUNGSI UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Memuat dataset dan mengembalikan dataframe."""
    df = pd.read_csv(DATASET_PATH)
    print(f"  Dataset: {os.path.basename(DATASET_PATH)}")
    print(f"  Observasi: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    return df


def run_evaluation(X_scaled, K_values, experiment_dir, exp_name):
    """Evaluasi K dari 2 sampai 10, simpan grafik, return best K."""
    inertia_values = []
    silhouette_values = []
    dbi_values = []

    for k in K_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia_values.append(km.inertia_)
        silhouette_values.append(silhouette_score(X_scaled, labels))
        dbi_values.append(davies_bouldin_score(X_scaled, labels))

    # Simpan tabel evaluasi
    eval_df = pd.DataFrame({
        "K": list(K_values),
        "Inertia": [f"{v:.2f}" for v in inertia_values],
        "Silhouette": [f"{v:.4f}" for v in silhouette_values],
        "DBI": [f"{v:.4f}" for v in dbi_values],
    })
    eval_df.to_csv(os.path.join(experiment_dir, "evaluasi_k.csv"), index=False)

    # Cari K optimal (Silhouette tertinggi)
    best_k_idx = np.argmax(silhouette_values)
    best_k = list(K_values)[best_k_idx]
    best_sil = silhouette_values[best_k_idx]
    best_dbi = dbi_values[best_k_idx]

    # Simpan grafik evaluasi
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(list(K_values), inertia_values, "o-", color="#2E86AB", linewidth=2, markersize=8)
    axes[0].set_title("Elbow Method (Inertia / WCSS)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Jumlah Klaster (K)", fontsize=11)
    axes[0].set_ylabel("Inertia", fontsize=11)
    axes[0].axvline(x=best_k, color="#E8505B", linestyle="--", linewidth=1.5, label=f"K optimal = {best_k}")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(K_values), silhouette_values, "s-", color="#F18F01", linewidth=2, markersize=8)
    axes[1].set_title("Silhouette Score per K", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Jumlah Klaster (K)", fontsize=11)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    axes[1].axvline(x=best_k, color="#E8505B", linestyle="--", linewidth=1.5, label=f"K optimal = {best_k}")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(list(K_values), dbi_values, "D-", color="#6B4226", linewidth=2, markersize=8)
    axes[2].set_title("Davies-Bouldin Index per K", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Jumlah Klaster (K)", fontsize=11)
    axes[2].set_ylabel("DBI", fontsize=11)
    axes[2].axvline(x=best_k, color="#E8505B", linestyle="--", linewidth=1.5, label=f"K optimal = {best_k}")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Evaluasi K Optimal — {exp_name}\n"
        f"Best K = {best_k} | Silhouette = {best_sil:.4f} | DBI = {best_dbi:.4f}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "evaluasi_k_optimal.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return best_k, best_sil, best_dbi, eval_df


def run_clustering(df, feature_names, feature_cols, K, experiment_dir, exp_name):
    """Jalankan K-Means++ dengan fitur dan K tertentu, simpan hasil."""
    print(f"\n{'='*60}")
    print(f"  {exp_name}")
    print(f"{'='*60}")
    print(f"  Fitur ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"  K = {K}")
    print()

    # Siapkan data
    X_raw = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Modelling
    kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Metrik
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    print(f"  Silhouette Score : {sil:.4f}")
    print(f"  DBI              : {dbi:.4f}")

    # Tambah label ke dataframe
    df_exp = df.copy()
    df_exp["Klaster"] = labels

    # Profiling
    profil = df_exp.groupby("Klaster")[feature_cols].mean()
    print(f"\n  Profil rata-rata per klaster:")
    print(f"  {profil.to_string()}")

    # Simpan hasil clustering
    output_cols = ["Provinsi", "Kab/Kota"] + feature_cols + ["Klaster"]
    df_exp[output_cols].to_csv(
        os.path.join(experiment_dir, "hasil_klaster.csv"), index=False
    )

    # Simpan profil
    profil.to_csv(os.path.join(experiment_dir, "profil_klaster.csv"))

    # Simpan ringkasan metrik
    with open(os.path.join(experiment_dir, "metrik.txt"), "w") as f:
        f.write(f"Eksperimen: {exp_name}\n")
        f.write(f"Fitur ({len(feature_names)}): {', '.join(feature_names)}\n")
        f.write(f"Jumlah Klaster (K): {K}\n")
        f.write(f"Jumlah Observasi: {X_scaled.shape[0]}\n")
        f.write(f"Silhouette Score: {sil:.4f}\n")
        f.write(f"Davies-Bouldin Index: {dbi:.4f}\n")

    return sil, dbi, labels, X_scaled


def plot_pca(df, feature_cols, labels, experiment_dir, exp_name, feature_names):
    """Buat PCA scatter plot 2D dari hasil clustering."""
    # Siapkan data
    X_raw = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_features = X_scaled.shape[1]
    n_components = min(2, max(1, n_features))

    if n_features == 1:
        # Kalau cuma 1 fitur, plot langsung (tidak perlu PCA)
        fig, ax = plt.subplots(figsize=(12, 8))
        pca_df = pd.DataFrame({
            "Fitur": X_scaled[:, 0],
            "Noise": np.random.normal(0, 0.02, size=len(X_scaled)),  # sedikit jitter
            "Klaster": labels.astype(str),
        })
        sns.scatterplot(
            data=pca_df,
            x="Fitur", y="Noise",
            hue="Klaster",
            palette=sns.color_palette("husl", len(set(labels))),
            s=60, alpha=0.7,
            edgecolor="white", linewidth=0.5,
            ax=ax,
        )
        ax.set_title(
            f"Distribusi 1D — {exp_name}\n"
            f"(Silhouette = {silhouette_score(X_scaled, labels):.4f})",
            fontsize=13, fontweight="bold", pad=15,
        )
        ax.set_xlabel(feature_names[0], fontsize=11)
        ax.set_ylabel("Jitter (untuk visualisasi)", fontsize=11)
        ax.set_yticks([])
        ax.legend(title="Klaster", fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "pca_scatter.png"), dpi=200, bbox_inches="tight")
        plt.close()
        return

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    total_var = sum(explained_var) * 100

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Klaster": labels.astype(str),
    })

    # Palet warna
    n_clusters = len(set(labels))
    palette = sns.color_palette("husl", n_clusters)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Klaster",
        palette=palette,
        s=60, alpha=0.7,
        edgecolor="white", linewidth=0.5,
        ax=ax,
    )

    ax.set_title(
        f"PCA Scatter Plot — {exp_name}\n"
        f"({total_var:.1f}% varians terekam | Silhouette = {silhouette_score(X_scaled, labels):.4f})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% varians)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% varians)", fontsize=11)
    ax.legend(title="Klaster", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "pca_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EKSEKUSI UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  EKSPERIMEN CLUSTERING — Polarisasi Kesejahteraan Indonesia")
    print("  Mencari konfigurasi fitur & K optimal untuk K-Means++")
    print("=" * 72)

    # Load data
    print("\n─── LOAD DATA ───")
    df = load_data()

    # Tabel untuk menyimpan hasil semua eksperimen
    all_results = []

    for exp in EXPERIMENTS:
        nama = exp["nama"]
        feature_names = exp["fitur"]
        feature_cols = get_feature_cols(feature_names)
        experiment_dir = os.path.join(EXPERIMEN_DIR, nama)

        # Hapus baris dengan missing values (kalau ada)
        df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
        print(f"\n  Data setelah cleaning: {df_clean.shape[0]} baris")

        # ── Step 1: Cari K optimal (khusus baseline) ──
        if exp["K_values"] is not None:
            print(f"\n  [EVALUASI K] Mencari K optimal dari {exp['K_values'][0]} s.d. {exp['K_values'][-1]}...")
            X_raw = df_clean[feature_cols].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            best_k, best_sil, best_dbi, eval_df = run_evaluation(
                X_scaled, exp["K_values"], experiment_dir, exp["description"]
            )
            print(f"  → K optimal: {best_k} (Silhouette={best_sil:.4f}, DBI={best_dbi:.4f})")

            # Simpan hasil evaluasi ke tabel
            all_results.append({
                "Eksperimen": nama,
                "Deskripsi": exp["description"],
                "Fitur": ", ".join(feature_names),
                "Jumlah Fitur": len(feature_names),
                "K": best_k,
                "Silhouette": best_sil,
                "DBI": best_dbi,
                "Metode K": "Optimal (evaluasi 2-10)",
                "Status": "✅ Evaluasi K",
            })

        # ── Step 2: Clustering dengan K tetap ──
        if exp["K_fixed"] is not None:
            K = exp["K_fixed"]
            sil, dbi, labels, X_scaled = run_clustering(
                df_clean, feature_names, feature_cols, K, experiment_dir, exp["description"]
            )

            # Simpan hasil ke tabel
            all_results.append({
                "Eksperimen": nama,
                "Deskripsi": exp["description"],
                "Fitur": ", ".join(feature_names),
                "Jumlah Fitur": len(feature_names),
                "K": K,
                "Silhouette": sil,
                "DBI": dbi,
                "Metode K": f"Tetap (K={K})",
                "Status": "✅ Selesai",
            })

            # PCA plot
            print(f"  [PCA] Membuat scatter plot...")
            plot_pca(df_clean, feature_cols, labels, experiment_dir, exp["description"], feature_names)

    # ─────────────────────────────────────────────────────────────────────────
    # TABEL PERBANDINGAN SEMUA EKSPERIMEN
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  📊 TABEL PERBANDINGAN SEMUA EKSPERIMEN")
    print("=" * 72)

    result_df = pd.DataFrame(all_results)

    # Urutkan: Silhouette terbaik di atas
    result_df = result_df.sort_values("Silhouette", ascending=False).reset_index(drop=True)
    result_df["Rank"] = range(1, len(result_df) + 1)
    result_df = result_df[["Rank", "Eksperimen", "Deskripsi", "Fitur", "Jumlah Fitur", "K", "Silhouette", "DBI", "Status"]]

    # Tampilkan
    print(f"\n{result_df.to_string(index=False)}")
    print()

    # Highlight top 3
    print("─" * 72)
    print("  🏆 TOP 3 EKSPERIMEN TERBAIK (berdasarkan Silhouette Score):")
    print("─" * 72)
    for i in range(min(3, len(result_df))):
        row = result_df.iloc[i]
        print(f"\n  #{row['Rank']} — {row['Eksperimen']}")
        print(f"     Fitur : {row['Fitur']}")
        print(f"     K     : {row['K']}")
        print(f"     Sil   : {row['Silhouette']:.4f}")
        print(f"     DBI   : {row['DBI']:.4f}")

    # Simpan ke CSV
    result_df.to_csv(os.path.join(EXPERIMEN_DIR, "perbandingan_eksperimen.csv"), index=False)
    print(f"\n\n  Tabel perbandingan disimpan di: eksperimen/perbandingan_eksperimen.csv")

    # ── Rekomendasi ──
    print("\n" + "=" * 72)
    print("  💡 REKOMENDASI")
    print("=" * 72)
    if len(result_df) >= 2:
        best = result_df.iloc[0]
        second = result_df.iloc[1]
        print(f"\n  Eksperimen TERBAIK: {best['Eksperimen']}")
        print(f"    • Silhouette = {best['Silhouette']:.4f}")
        print(f"    • DBI        = {best['DBI']:.4f}")
        print(f"    • Fitur      : {best['Fitur']}")
        print(f"    • K          : {best['K']}")

        if abs(best["Silhouette"] - second["Silhouette"]) > 0.05:
            print(f"\n  📌 Eksperimen ini SIGNIFIKAN lebih baik dari lainnya.")
        else:
            print(f"\n  📌 Selisih dengan eksperimen lain tipis. Bisa dipilih yang paling")
            print(f"     masuk akal secara interpretasi (domain knowledge).")

    print("\n" + "=" * 72)
    print("  EKSPERIMEN SELESAI! 🎉")
    print("=" * 72)


if __name__ == "__main__":
    main()
