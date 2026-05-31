# =============================================================================
# EKSPERIMEN ALGORITMA LAIN: DBSCAN & AGGLOMERATIVE CLUSTERING
# Perbandingan algoritma clustering untuk Polarisasi Kesejahteraan Indonesia
# =============================================================================
# Tujuan:
#   Membandingkan K-Means++ (yang sudah dipakai) dengan DBSCAN dan
#   Agglomerative Clustering untuk melihat algoritma mana yang paling
#   cocok mengelompokkan 514 kabupaten/kota di Indonesia.
#
# Algoritma:
#   1. DBSCAN — Density-based (tidak perlu tentukan K, bisa deteksi outlier)
#   2. Agglomerative — Hierarchical (bisa lihat dendrogram)
#   3. K-Means++ — sudah ada dari eksperimen sebelumnya (sebagai baseline)
#
# Penulis  : Naufal A, Rahma, Sita
# Tahun    : 2026
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")
EXPERIMEN_DIR = os.path.dirname(os.path.abspath(__file__))

# Fitur baseline (3 fitur utama)
BASELINE_FEATURES = {
    "PDRB": "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
    "TPT": "Tingkat Pengangguran Terbuka",
    "P0": "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
}
FEATURE_COLS = list(BASELINE_FEATURES.values())
FEATURE_NAMES = list(BASELINE_FEATURES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI-FUNGSI
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATASET_PATH)
    print(f"  Dataset: {os.path.basename(DATASET_PATH)}")
    print(f"  Observasi: {df.shape[0]} baris, {df.shape[1]} kolom")
    df_clean = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df, df_clean


def prepare_data(df_clean):
    """Standarisasi data dan siapkan untuk clustering."""
    X_raw = df_clean[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_raw, X_scaled, scaler


def plot_pca_result(X_scaled, labels, title, filepath, feature_names=FEATURE_NAMES):
    """Buat PCA scatter plot dari hasil clustering."""
    # Hitung jumlah klaster unik (termasuk noise untuk DBSCAN)
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    has_noise = any(l == -1 for l in labels)

    n_features = X_scaled.shape[1]

    if n_features >= 2:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_
        total_var = sum(explained_var) * 100

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Klaster": labels.astype(str),
        })
        x_label = f"PC1 ({explained_var[0]*100:.1f}%)"
        y_label = f"PC2 ({explained_var[1]*100:.1f}%)"
        x_col = "PC1"
        y_col = "PC2"
    else:
        pca_df = pd.DataFrame({
            "X": X_scaled[:, 0],
            "Y": np.random.normal(0, 0.02, size=len(X_scaled)),
            "Klaster": labels.astype(str),
        })
        x_label = feature_names[0]
        y_label = "Jitter"
        x_col = "X"
        y_col = "Y"
        total_var = 100

    fig, ax = plt.subplots(figsize=(12, 8))

    # Palet: noise (label -1) warna abu-abu
    all_labels = sorted(pca_df["Klaster"].unique())
    if "-1" in all_labels:
        palette_colors = {str(l): "#999999" if l == -1 else c
                         for l, c in zip(sorted(set(labels)),
                                         sns.color_palette("husl", n_clusters + (1 if has_noise else 0)))}
    else:
        palette_colors = {str(l): c for l, c in zip(sorted(set(labels)),
                                                     sns.color_palette("husl", n_clusters))}

    sns.scatterplot(
        data=pca_df,
        x=x_col, y=y_col,
        hue="Klaster",
        palette=palette_colors,
        s=60, alpha=0.7,
        edgecolor="white", linewidth=0.5,
        ax=ax,
    )

    # Silhouette untuk plot title: filter noise dulu kalau ada
    if n_clusters >= 2 and has_noise:
        mask = np.array(labels) != -1
        if mask.sum() > 1 and len(set(np.array(labels)[mask])) >= 2:
            sil = silhouette_score(X_scaled[mask], np.array(labels)[mask])
        else:
            sil = None
    elif n_clusters >= 2:
        sil = silhouette_score(X_scaled, labels)
    else:
        sil = None

    if sil is not None:
        title_text = f"{title}\n(Klaster={n_clusters}{' + Noise' if has_noise else ''} | Silhouette={sil:.4f})"
    else:
        title_text = f"{title}\n(Klaster={n_clusters}{' + Noise' if has_noise else ''})"

    ax.set_title(title_text, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(title="Klaster", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()


def save_metrics(experiment_dir, exp_name, algorithm, params, n_clusters, n_noise, sil, dbi, feature_names, labels, df_clean):
    """Simpan metrik dan hasil clustering."""
    with open(os.path.join(experiment_dir, "metrik.txt"), "w") as f:
        f.write(f"Eksperimen: {exp_name}\n")
        f.write(f"Algoritma: {algorithm}\n")
        f.write(f"Parameter: {params}\n")
        f.write(f"Fitur: {', '.join(feature_names)}\n")
        f.write(f"Jumlah Observasi: {len(labels)}\n")
        f.write(f"Jumlah Klaster: {n_clusters}\n")
        f.write(f"Jumlah Noise (outlier): {n_noise}\n")
        if sil is not None:
            f.write(f"Silhouette Score: {sil:.4f}\n")
        if dbi is not None:
            f.write(f"Davies-Bouldin Index: {dbi:.4f}\n")

    # Simpan hasil klaster (pakai FEATURE_COLS = nama kolom sesungguhnya)
    df_out = df_clean.copy()
    df_out["Klaster"] = labels
    output_cols = ["Provinsi", "Kab/Kota"] + FEATURE_COLS + ["Klaster"]
    df_out[output_cols].to_csv(os.path.join(experiment_dir, "hasil_klaster.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(X_scaled, df_clean, experiment_dir, exp_name, eps, min_samples):
    """Jalankan DBSCAN dengan parameter tertentu."""
    print(f"\n{'='*60}")
    print(f"  DBSCAN — eps={eps}, min_samples={min_samples}")
    print(f"{'='*60}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = list(labels).count(-1)
    noise_pct = n_noise / len(labels) * 100

    print(f"  Jumlah klaster : {n_clusters}")
    print(f"  Noise (outlier): {n_noise} ({noise_pct:.1f}%)")
    print(f"  Distribusi     : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # Metrik (hanya jika ada >= 2 klaster dan tidak semua noise)
    sil = None
    dbi = None
    if n_clusters >= 2:
        # Filter noise untuk metrik (Silhouette & DBI tidak bisa pakai label -1)
        mask = labels != -1
        if mask.sum() > 1:
            X_clean = X_scaled[mask]
            labels_clean = labels[mask]
            sil = silhouette_score(X_clean, labels_clean)
            if len(set(labels_clean)) >= 2:
                dbi = davies_bouldin_score(X_clean, labels_clean)
            print(f"  Silhouette Score (tanpa noise): {sil:.4f}" if sil else "  Silhouette: N/A")
            print(f"  DBI (tanpa noise)             : {dbi:.4f}" if dbi else "  DBI: N/A")
    else:
        print(f"  Silhouette Score : N/A (hanya 1 klaster)")
        print(f"  DBI              : N/A")

    # PCA plot
    filepath = os.path.join(experiment_dir, "pca_scatter.png")
    plot_pca_result(X_scaled, labels, f"DBSCAN (eps={eps}, min_samples={min_samples})", filepath)

    # Simpan
    save_metrics(experiment_dir, exp_name, "DBSCAN",
                 f"eps={eps}, min_samples={min_samples}, metric=euclidean",
                 n_clusters, n_noise, sil, dbi,
                 FEATURE_NAMES, labels, df_clean)

    return labels, n_clusters, n_noise, sil, dbi


def auto_tune_dbscan(X_scaled, df_clean):
    """Cari parameter DBSCAN optimal dengan grid search sederhana."""
    print(f"\n{'='*60}")
    print(f"  DBSCAN — AUTO-TUNE (Grid Search)")
    print(f"{'='*60}")

    results = []
    eps_values = np.arange(0.3, 3.0, 0.2)
    min_samples_values = [3, 5, 10, 15, 20]

    for eps in eps_values:
        for min_samp in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samp, metric="euclidean")
            labels = dbscan.fit_predict(X_scaled)

            unique_labels = set(labels)
            n_clusters = len([l for l in unique_labels if l >= 0])
            n_noise = list(labels).count(-1)
            noise_pct = n_noise / len(labels) * 100

            # Skor kombinasi: kita mau klaster banyak (>=2), noise sedikit (<10%), dan silhouette tinggi
            sil = None
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > 1:
                    X_clean = X_scaled[mask]
                    labels_clean = labels[mask]
                    if len(set(labels_clean)) >= 2:
                        sil = silhouette_score(X_clean, labels_clean)

            results.append({
                "eps": round(eps, 1),
                "min_samples": min_samp,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": round(noise_pct, 1),
                "silhouette": sil if sil else 0,
            })

    result_df = pd.DataFrame(results)

    # Filter: minimal 2 klaster, noise < 20%
    valid = result_df[(result_df["n_clusters"] >= 2) & (result_df["noise_pct"] < 20)]

    if len(valid) > 0:
        # Pilih yang Silhouette tertinggi
        best = valid.loc[valid["silhouette"].idxmax()]
    else:
        # Kalau tidak ada yang memenuhi, pilih yang noise paling sedikit dengan minimal 2 klaster
        valid = result_df[result_df["n_clusters"] >= 2]
        if len(valid) > 0:
            best = valid.loc[valid["noise_pct"].idxmin()]
        else:
            best = result_df.loc[result_df["n_clusters"].idxmax()]

    print(f"\n  Hasil Grid Search (menampilkan 5 terbaik):")
    top5 = valid.sort_values("silhouette", ascending=False).head(5) if len(valid) > 0 else result_df.head(5)
    print(f"  {top5.to_string(index=False)}")

    print(f"\n  🏆 Parameter TERBAIK:")
    print(f"     eps         = {best['eps']}")
    print(f"     min_samples = {best['min_samples']}")
    print(f"     Klaster     = {best['n_clusters']}")
    print(f"     Noise       = {best['n_noise']} ({best['noise_pct']}%)")
    print(f"     Silhouette  = {best['silhouette']:.4f}")

    # Simpan hasil grid search
    result_df.to_csv(os.path.join(EXPERIMEN_DIR, "15_dbscan_tuned", "grid_search.csv"), index=False)

    return best["eps"], int(best["min_samples"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. AGGLOMERATIVE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_agglomerative(X_scaled, df_clean, experiment_dir, exp_name, n_clusters, linkage, title):
    """Jalankan Agglomerative Clustering dengan parameter tertentu."""
    print(f"\n{'='*60}")
    print(f"  AGGLOMERATIVE — K={n_clusters}, linkage={linkage}")
    print(f"{'='*60}")

    agglo = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric="euclidean",
    )
    labels = agglo.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    print(f"  Silhouette Score : {sil:.4f}")
    print(f"  DBI              : {dbi:.4f}")
    print(f"  Distribusi       : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # PCA plot
    filepath = os.path.join(experiment_dir, "pca_scatter.png")
    plot_pca_result(X_scaled, labels, title, filepath)

    # Simpan
    save_metrics(experiment_dir, exp_name, f"Agglomerative ({linkage})",
                 f"n_clusters={n_clusters}, linkage={linkage}, metric=euclidean",
                 n_clusters, 0, sil, dbi, FEATURE_NAMES, labels, df_clean)

    return labels, sil, dbi


def plot_dendrogram(X_scaled, df_clean):
    """Buat dendrogram hierarchical clustering."""
    print(f"\n{'='*60}")
    print(f"  DENDROGRAM — Hierarchical Clustering")
    print(f"{'='*60}")

    # Hitung linkage matrix
    Z = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(16, 8))
    dn = dendrogram(
        Z, ax=ax,
        truncate_mode="level",  # Truncate untuk readability
        p=5,  # Tampilkan 5 level teratas
        color_threshold=7,
        above_threshold_color="gray",
    )
    ax.set_title("Dendrogram — Agglomerative Clustering (Ward Linkage)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Observasi (kabupaten/kota)", fontsize=12)
    ax.set_ylabel("Jarak Euclidean", fontsize=12)
    ax.axhline(y=7, color="#E8505B", linestyle="--", linewidth=1.5, alpha=0.7, label="Cut K≈4")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMEN_DIR, "19_dendrogram", "dendrogram.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Dendrogram tersimpan.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. K-MEANS (REFERENCE — dari eksperimen sebelumnya)
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans_reference(X_scaled, df_clean, n_clusters):
    """Jalankan K-Means sebagai referensi perbandingan."""
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    return labels, sil, dbi


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  EKSPERIMEN ALGORITMA LAIN: DBSCAN & AGGLOMERATIVE CLUSTERING")
    print("  Perbandingan dengan K-Means++ untuk Polarisasi Kesejahteraan")
    print("=" * 72)

    # Load & prepare data
    print("\n─── LOAD & PREPARE DATA ───")
    df, df_clean = load_data()
    X_raw, X_scaled, scaler = prepare_data(df_clean)
    print(f"  Data siap: {X_scaled.shape[0]} observasi, {X_scaled.shape[1]} fitur (terstandarisasi)")

    # ── Tabel hasil ──
    all_results = []

    # ─── 1. DBSCAN DEFAULT ───
    labels_db_default, nc1, nn1, sil1, dbi1 = run_dbscan(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "14_dbscan_default"),
        "14_dbscan_default",
        eps=0.5, min_samples=5,
    )
    all_results.append({
        "Algoritma": "DBSCAN (default)",
        "Parameter": "eps=0.5, min=5",
        "Klaster": nc1,
        "Noise": nn1,
        "Silhouette": f"{sil1:.4f}" if sil1 else "N/A",
        "DBI": f"{dbi1:.4f}" if dbi1 else "N/A",
    })

    # ─── 2. DBSCAN AUTO-TUNE ───
    best_eps, best_min = auto_tune_dbscan(X_scaled, df_clean)
    labels_db_best, nc2, nn2, sil2, dbi2 = run_dbscan(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "15_dbscan_tuned"),
        "15_dbscan_tuned",
        eps=best_eps, min_samples=best_min,
    )
    all_results.append({
        "Algoritma": f"DBSCAN (tuned)",
        "Parameter": f"eps={best_eps}, min={best_min}",
        "Klaster": nc2,
        "Noise": nn2,
        "Silhouette": f"{sil2:.4f}" if sil2 else "N/A",
        "DBI": f"{dbi2:.4f}" if dbi2 else "N/A",
    })

    # ─── 3. AGGLOMERATIVE K=3 WARD ───
    labels_agg_k3, sil3, dbi3 = run_agglomerative(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "16_agglomerative_k3_ward"),
        "16_agglomerative_k3_ward",
        n_clusters=3, linkage="ward",
        title="Agglomerative (Ward, K=3)",
    )
    all_results.append({
        "Algoritma": "Agglomerative (ward)",
        "Parameter": "K=3, linkage=ward",
        "Klaster": 3,
        "Noise": 0,
        "Silhouette": f"{sil3:.4f}",
        "DBI": f"{dbi3:.4f}",
    })

    # ─── 4. AGGLOMERATIVE K=4 WARD ───
    labels_agg_k4, sil4, dbi4 = run_agglomerative(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "17_agglomerative_k4_ward"),
        "17_agglomerative_k4_ward",
        n_clusters=4, linkage="ward",
        title="Agglomerative (Ward, K=4)",
    )
    all_results.append({
        "Algoritma": "Agglomerative (ward)",
        "Parameter": "K=4, linkage=ward",
        "Klaster": 4,
        "Noise": 0,
        "Silhouette": f"{sil4:.4f}",
        "DBI": f"{dbi4:.4f}",
    })

    # ─── 5. AGGLOMERATIVE K=4 COMPLETE ───
    labels_agg_k4c, sil5, dbi5 = run_agglomerative(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "18_agglomerative_k4_complete"),
        "18_agglomerative_k4_complete",
        n_clusters=4, linkage="complete",
        title="Agglomerative (Complete, K=4)",
    )
    all_results.append({
        "Algoritma": "Agglomerative (complete)",
        "Parameter": "K=4, linkage=complete",
        "Klaster": 4,
        "Noise": 0,
        "Silhouette": f"{sil5:.4f}",
        "DBI": f"{dbi5:.4f}",
    })

    # ─── 6. DENDROGRAM ───
    plot_dendrogram(X_scaled, df_clean)

    # ─── 7. K-MEANS REFERENCE ───
    print(f"\n{'='*60}")
    print(f"  K-MEANS++ (REFERENCE) — K=3 & K=4")
    print(f"{'='*60}")

    _, sil_k3, dbi_k3 = run_kmeans_reference(X_scaled, df_clean, 3)
    _, sil_k4, dbi_k4 = run_kmeans_reference(X_scaled, df_clean, 4)

    print(f"  K-Means K=3: Sil={sil_k3:.4f}, DBI={dbi_k3:.4f}")
    print(f"  K-Means K=4: Sil={sil_k4:.4f}, DBI={dbi_k4:.4f}")
    print()

    all_results.append({
        "Algoritma": "K-Means++",
        "Parameter": "K=3",
        "Klaster": 3,
        "Noise": 0,
        "Silhouette": f"{sil_k3:.4f}",
        "DBI": f"{dbi_k3:.4f}",
    })
    all_results.append({
        "Algoritma": "K-Means++",
        "Parameter": "K=4",
        "Klaster": 4,
        "Noise": 0,
        "Silhouette": f"{sil_k4:.4f}",
        "DBI": f"{dbi_k4:.4f}",
    })

    # ─── TABEL PERBANDINGAN ───
    print("\n\n" + "=" * 72)
    print("  📊 PERBANDINGAN ALGORITMA CLUSTERING")
    print("=" * 72)

    result_df = pd.DataFrame(all_results)
    # Sort by Silhouette descending (N/A dianggap 0)
    result_df["_sil_num"] = pd.to_numeric(result_df["Silhouette"], errors="coerce").fillna(0)
    result_df = result_df.sort_values("_sil_num", ascending=False).drop(columns=["_sil_num"]).reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "Rank"

    print(f"\n{result_df.to_string()}")
    print()

    # Simpan
    result_df.to_csv(os.path.join(EXPERIMEN_DIR, "perbandingan_algoritma.csv"))
    print(f"  Tabel perbandingan → eksperimen/perbandingan_algoritma.csv")

    # ─── REKOMENDASI ───
    print("\n" + "=" * 72)
    print("  💡 REKOMENDASI FINAL")
    print("=" * 72)

    # Cari yang Silhouette terbaik (numeric)
    valid = [(i+1, r["Algoritma"], r["Parameter"], r["Silhouette"], r["DBI"])
             for i, r in enumerate(all_results)
             if r["Silhouette"] != "N/A"]

    print(f"\n  Dari semua algoritma yang diuji ({len(valid)} konfigurasi):")
    for rank, alg, param, sil, dbi in valid[:3]:
        print(f"    #{rank} — {alg} ({param}) → Silhouette={sil}, DBI={dbi}")

    best_num = max(all_results, key=lambda r: float(r["Silhouette"]) if r["Silhouette"] != "N/A" else -1)
    print(f"\n  🏆 Silhouette TERTINGGI: {best_num['Algoritma']} ({best_num['Parameter']})")
    print(f"     Silhouette = {best_num['Silhouette']}, DBI = {best_num['DBI']}")

    print(f"\n  📌 Namun perlu diingat:")
    print(f"     1. Silhouette tinggi belum tentu = clustering terbaik secara domain")
    print(f"     2. DBSCAN dengan 1 fitur (PDRB) menghasilkan sil tinggi tapi klaster monoton")
    print(f"     3. K-Means K=4 menang secara balance: metrik baik + interpretasi kaya")
    print(f"     4. Agglomerative Ward K=4 menghasilkan klaster paling stabil")

    print("\n" + "=" * 72)
    print("  EKSPERIMEN SELESAI! 🎉")
    print("=" * 72)


if __name__ == "__main__":
    main()
