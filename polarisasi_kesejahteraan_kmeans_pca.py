# =============================================================================
# PROGRAM KOMPUTER: POLARISASI KESEJAHTERAAN DAN KEJENUHAN WILAYAH
# Klasterisasi Paradoks Pertumbuhan Ekonomi Menggunakan K-Means++ dan PCA
# =============================================================================
# Deskripsi:
#   Program ini mendeteksi polarisasi spasial (Paradoks Industrialisasi/
#   Kejenuhan Ekonomi) pada tingkat kabupaten/kota di Indonesia. Anomali
#   episentrum ekonomi nasional dengan PDRB raksasa yang justru menjadi
#   kantong pengangguran tinggi diidentifikasi melalui klasterisasi
#   unsupervised K-Means++ dan reduksi dimensi PCA.
#
# Penulis  : Naufal A, Rahma, Sita
# Tahun    : 2026
# Lisensi  : Hak Kekayaan Intelektual – Program Komputer
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 0: IMPOR PUSTAKA (LIBRARY IMPORTS)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Konfigurasi visualisasi agar menghasilkan grafik beresolusi tinggi
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "serif"
sns.set_style("whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 1: DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
# Memuat dataset Klasifikasi Tingkat Kemiskinan di Indonesia
import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(PROJECT_DIR, "dataset", "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")
df = pd.read_csv(FILE_PATH)

print("=" * 72)
print("BAGIAN 1: DATA LOADING & PREPROCESSING")
print("=" * 72)
print(f"Jumlah observasi awal           : {df.shape[0]} baris")
print(f"Jumlah variabel awal            : {df.shape[1]} kolom")
print()

# Mendefinisikan tiga fitur utama yang menjadi basis analisis klasterisasi
FEATURE_COLS = [
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
    "Tingkat Pengangguran Terbuka",
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
]

# Memeriksa dan menghapus nilai hilang (missing values) pada kolom fitur
missing_before = df[FEATURE_COLS].isnull().sum().sum()
print(f"Total missing values pada fitur  : {missing_before}")
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"Jumlah observasi setelah cleaning: {df.shape[0]} baris")
print()

# Mengekstraksi matriks fitur asli (sebelum standarisasi) untuk profiling
X_raw = df[FEATURE_COLS].copy()

# Standarisasi menggunakan StandardScaler (z-score normalization)
# Langkah ini wajib karena skala antar-fitur sangat berbeda:
#   - PDRB berskala triliunan Rupiah
#   - TPT dan Kemiskinan berskala persentase (0-100)
# Tanpa standarisasi, K-Means akan didominasi oleh fitur berskala besar.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Statistik deskriptif setelah standarisasi (mean ≈ 0, std ≈ 1):")
print(pd.DataFrame(X_scaled, columns=FEATURE_COLS).describe().round(4))
print()

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 2: PENENTUAN K OPTIMAL & EVALUASI UNSUPERVISED
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BAGIAN 2: PENENTUAN K OPTIMAL (ELBOW METHOD & SILHOUETTE SCORE)")
print("=" * 72)

K_RANGE = range(2, 11)  # Evaluasi K dari 2 hingga 10
inertia_values = []
silhouette_values = []
dbi_values = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia_values.append(km.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels))
    dbi_values.append(davies_bouldin_score(X_scaled, labels))

# Menampilkan tabel evaluasi metrik untuk setiap nilai K
eval_df = pd.DataFrame({
    "K": list(K_RANGE),
    "Inertia (WCSS)": inertia_values,
    "Silhouette Score": silhouette_values,
    "Davies-Bouldin Index": dbi_values,
})
print(eval_df.to_string(index=False))
print()

# Visualisasi Elbow Method (Inertia / Within-Cluster Sum of Squares)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Elbow Curve
axes[0].plot(list(K_RANGE), inertia_values, "o-", color="#2E86AB", linewidth=2, markersize=8)
axes[0].set_title("Elbow Method (Inertia / WCSS)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Jumlah Klaster (K)", fontsize=11)
axes[0].set_ylabel("Inertia", fontsize=11)
axes[0].axvline(x=3, color="#E8505B", linestyle="--", linewidth=1.5, label="K = 3 (dipilih)")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Silhouette Score
axes[1].plot(list(K_RANGE), silhouette_values, "s-", color="#F18F01", linewidth=2, markersize=8)
axes[1].set_title("Silhouette Score per K", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Jumlah Klaster (K)", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].axvline(x=3, color="#E8505B", linestyle="--", linewidth=1.5, label="K = 3 (dipilih)")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Subplot 3: Davies-Bouldin Index
axes[2].plot(list(K_RANGE), dbi_values, "D-", color="#6B4226", linewidth=2, markersize=8)
axes[2].set_title("Davies-Bouldin Index per K", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Jumlah Klaster (K)", fontsize=11)
axes[2].set_ylabel("DBI", fontsize=11)
axes[2].axvline(x=3, color="#E8505B", linestyle="--", linewidth=1.5, label="K = 3 (dipilih)")
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle(
    "Evaluasi Penentuan Jumlah Klaster Optimal (K = 2 s.d. 10)",
    fontsize=15, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("Evaluasi_K_Optimal.png", dpi=200, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 3: MODELLING K-MEANS++ (n_clusters=3)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BAGIAN 3: MODELLING K-MEANS++ (K = 3)")
print("=" * 72)

K_OPTIMAL = 3
kmeans_model = KMeans(
    n_clusters=K_OPTIMAL,
    init="k-means++",
    n_init=10,
    random_state=42,
)
cluster_labels = kmeans_model.fit_predict(X_scaled)

# Menambahkan label klaster ke dataset asli
df["Klaster"] = cluster_labels

print(f"Model K-Means++ berhasil di-fitting pada {X_scaled.shape[0]} observasi.")
print(f"Distribusi observasi per klaster:")
print(df["Klaster"].value_counts().sort_index().to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 4: EVALUASI MODEL (PASCA-PEMROSESAN)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BAGIAN 4: EVALUASI MODEL – METRIK UNSUPERVISED (K = 3)")
print("=" * 72)

sil_score = silhouette_score(X_scaled, cluster_labels)
dbi_score = davies_bouldin_score(X_scaled, cluster_labels)

print(f"  Silhouette Score   : {sil_score:.4f}")
print(f"  Davies-Bouldin Index (DBI): {dbi_score:.4f}")
print()
print("─" * 72)
print("INTERPRETASI METRIK EVALUASI:")
print("─" * 72)
print(
    "  • Silhouette Score mengukur seberapa kohesif suatu observasi berada\n"
    "    di dalam klasternya dibandingkan klaster terdekat. Nilai berkisar\n"
    "    antara -1 hingga +1. Semakin tinggi (mendekati 1), semakin baik\n"
    "    separasi antar-klaster, yang berarti model berhasil memisahkan\n"
    "    polarisasi wilayah secara distingtif.\n"
)
print(
    "  • Davies-Bouldin Index (DBI) mengukur rata-rata rasio dispersi\n"
    "    intra-klaster terhadap jarak antar-pusat klaster. Semakin rendah\n"
    "    (mendekati 0), semakin baik pemisahan klaster. Nilai DBI rendah\n"
    "    mengindikasikan bahwa klaster-klaster yang terbentuk kompak secara\n"
    "    internal dan terpisah secara spasial – mendukung hipotesis\n"
    "    polarisasi kesejahteraan yang signifikan antar-tipologi wilayah.\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 5: VISUALISASI PCA (PRINCIPAL COMPONENT ANALYSIS)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BAGIAN 5: REDUKSI DIMENSI PCA & VISUALISASI SCATTER PLOT 2D")
print("=" * 72)

# Mereduksi 3 dimensi fitur terstandarisasi menjadi 2 komponen utama
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Menampilkan proporsi varians yang dijelaskan oleh masing-masing komponen
explained_var = pca.explained_variance_ratio_
print(f"  Varians dijelaskan PC1: {explained_var[0]:.4f} ({explained_var[0]*100:.2f}%)")
print(f"  Varians dijelaskan PC2: {explained_var[1]:.4f} ({explained_var[1]*100:.2f}%)")
print(f"  Total varians terekam : {sum(explained_var):.4f} ({sum(explained_var)*100:.2f}%)")
print()

# Membangun DataFrame untuk visualisasi
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Klaster": cluster_labels.astype(str),
    "Kab/Kota": df["Kab/Kota"].values,
})

# Mendefinisikan palet warna yang distingtif untuk setiap klaster
CLUSTER_PALETTE = {"0": "#2E86AB", "1": "#E8505B", "2": "#F18F01"}

# Membuat Scatter Plot PCA 2D yang elegan menggunakan Seaborn
fig, ax = plt.subplots(figsize=(14, 9))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Klaster",
    palette=CLUSTER_PALETTE,
    s=80,
    alpha=0.75,
    edgecolor="white",
    linewidth=0.5,
    ax=ax,
)

# Anotasi kutub wilayah anomali industri (episentrum ekonomi jenuh)
# dan kutub kerentanan struktural
ANNOTATION_TARGETS = [
    "Karawang", "Kota Cilegon", "Bogor", "Intan Jaya",
    "Kota Surabaya", "Kota Jakarta Pusat", "Kota Bekasi",
    "Kota Bandung", "Kota Makassar",
    "Nduga *", "Deiyai", "Lanny Jaya",
]

for _, row in pca_df.iterrows():
    if row["Kab/Kota"] in ANNOTATION_TARGETS:
        ax.annotate(
            row["Kab/Kota"],
            xy=(row["PC1"], row["PC2"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.85),
        )

ax.set_title(
    "Visualisasi Klasterisasi Polarisasi Kesejahteraan Wilayah\n"
    "(Proyeksi PCA 2 Komponen Utama – K-Means++ K=3)",
    fontsize=14, fontweight="bold", pad=15,
)
ax.set_xlabel(f"Principal Component 1 ({explained_var[0]*100:.2f}% varians)", fontsize=12)
ax.set_ylabel(f"Principal Component 2 ({explained_var[1]*100:.2f}% varians)", fontsize=12)
ax.legend(title="Klaster", fontsize=10, title_fontsize=11, loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Visualisasi_PCA_Klaster_Polarisasi.png", dpi=200, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 6: PROFILING KLASTER & EKSPOR DATA FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("BAGIAN 6: PROFILING KLASTER & PELABELAN TIPOLOGI POLARISASI")
print("=" * 72)

# Menghitung nilai rata-rata asli (belum di-scale) untuk setiap klaster
RENAME_MAP = {
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)": "Rata-rata PDRB (Rp)",
    "Tingkat Pengangguran Terbuka": "Rata-rata TPT (%)",
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)": "Rata-rata Kemiskinan (%)",
}

profil_klaster = (
    df.groupby("Klaster")[FEATURE_COLS]
    .mean()
    .rename(columns=RENAME_MAP)
)
print("\nProfil Agregasi Nilai Rata-rata per Klaster (Data Asli):")
print(profil_klaster.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# Pelabelan Tipologi Polarisasi secara otomatis berdasarkan profil klaster
# ─────────────────────────────────────────────────────────────────────────────
# Logika pelabelan:
#   - Klaster dengan PDRB tertinggi → "Episentrum Ekonomi Jenuh / Rentan"
#     (paradoks: PDRB tinggi namun TPT juga cenderung tinggi)
#   - Klaster dengan Kemiskinan tertinggi → "Kerentanan Struktural"
#     (wilayah terbelakang secara ekonomi dan sosial)
#   - Klaster sisanya → "Wilayah Transisi / Berkembang"

pdrb_col = "Rata-rata PDRB (Rp)"
miskin_col = "Rata-rata Kemiskinan (%)"

klaster_pdrb_tertinggi = profil_klaster[pdrb_col].idxmax()
klaster_miskin_tertinggi = profil_klaster[miskin_col].idxmax()
klaster_transisi = [
    k for k in profil_klaster.index
    if k != klaster_pdrb_tertinggi and k != klaster_miskin_tertinggi
][0]

TIPOLOGI_MAP = {
    klaster_pdrb_tertinggi: "Episentrum Ekonomi Jenuh / Rentan",
    klaster_miskin_tertinggi: "Kerentanan Struktural",
    klaster_transisi: "Wilayah Transisi / Berkembang",
}

print("Pemetaan Label Tipologi Polarisasi:")
for klaster_id, tipologi in sorted(TIPOLOGI_MAP.items()):
    print(f"  Klaster {klaster_id} → {tipologi}")
print()

# Menambahkan kolom tipologi ke dataset
df["Tipologi Polarisasi"] = df["Klaster"].map(TIPOLOGI_MAP)

# Menampilkan contoh wilayah per tipologi
print("─" * 72)
print("CONTOH WILAYAH REPRESENTATIF PER TIPOLOGI:")
print("─" * 72)
for tipologi_name in TIPOLOGI_MAP.values():
    subset = df[df["Tipologi Polarisasi"] == tipologi_name]
    sample_regions = subset["Kab/Kota"].head(5).tolist()
    print(f"\n  [{tipologi_name}]")
    print(f"  Jumlah wilayah: {len(subset)}")
    print(f"  Contoh        : {', '.join(sample_regions)}")
print()

# Menampilkan profil klaster dengan label tipologi
profil_final = profil_klaster.copy()
profil_final["Tipologi Polarisasi"] = profil_final.index.map(TIPOLOGI_MAP)
print("\nProfil Agregasi Final dengan Tipologi:")
print(profil_final.to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# Ekspor dataset final ke file CSV
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "Hasil_Klaster_Polarisasi_Kesejahteraan.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset final berhasil diekspor ke: {OUTPUT_PATH}")
print(f"Total baris diekspor: {df.shape[0]}")
print(f"Kolom tambahan      : 'Klaster', 'Tipologi Polarisasi'")
print()
print("=" * 72)
print("PIPELINE ANALISIS KLASTERISASI POLARISASI SELESAI.")
print("=" * 72)
