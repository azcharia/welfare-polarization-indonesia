# =============================================================================
# EKSPERIMEN ALGORITMA LANJUTAN: GMM, SPECTRAL, MEAN SHIFT, OPTICS
# Perbandingan clustering untuk Polarisasi Kesejahteraan Indonesia
# =============================================================================
# Algoritma baru:
#   1. GMM          — Gaussian Mixture Model (soft clustering + probabilitas)
#   2. Spectral     — Graph-based clustering
#   3. Mean Shift   — Density mode-seeking (auto-detect K)
#   4. OPTICS       — Improved DBSCAN (varying densities)
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_DIR, "dataset", "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")
EXPERIMEN_DIR = os.path.dirname(os.path.abspath(__file__))

BASELINE_FEATURES = {
    "PDRB": "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
    "TPT": "Tingkat Pengangguran Terbuka",
    "P0": "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
}
FEATURE_COLS = list(BASELINE_FEATURES.values())
FEATURE_NAMES = list(BASELINE_FEATURES.keys())
N_FEATURES = len(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI BANTU
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df_clean = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df, df_clean

def prepare_data(df_clean):
    X_raw = df_clean[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_raw, X_scaled, scaler

def count_clusters(labels):
    """Hitung jumlah klaster unik (abaikan -1 = noise)."""
    return len([l for l in set(labels) if l >= 0])

def plot_pca_result(X_scaled, labels, title, filepath, palette_name="husl"):
    """Buat PCA scatter plot 2D dari hasil clustering."""
    unique_labels = sorted(set(labels))
    n_clusters = count_clusters(labels)
    has_noise = any(l == -1 for l in labels)
    n_features = X_scaled.shape[1]

    if n_features >= 2:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_
        pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1], "Klaster": labels.astype(str)})
        x_label, y_label = f"PC1 ({explained_var[0]*100:.1f}%)", f"PC2 ({explained_var[1]*100:.1f}%)"
        x_col, y_col = "PC1", "PC2"
    else:
        pca_df = pd.DataFrame({
            "X": X_scaled[:,0],
            "Y": np.random.normal(0, 0.02, size=len(X_scaled)),
            "Klaster": labels.astype(str),
        })
        x_label, y_label = FEATURE_NAMES[0], "Jitter"
        x_col, y_col = "X", "Y"

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette(palette_name, n_clusters + (1 if has_noise else 0))
    palette_colors = {}
    sorted_u = sorted(set(labels))
    ci = 0
    for l in sorted_u:
        if l == -1:
            palette_colors[str(l)] = "#999999"
        else:
            palette_colors[str(l)] = palette[ci]
            ci += 1

    sns.scatterplot(data=pca_df, x=x_col, y=y_col, hue="Klaster",
                    palette=palette_colors, s=60, alpha=0.7,
                    edgecolor="white", linewidth=0.5, ax=ax)

    # Hitung Silhouette dengan benar (filter noise)
    sil_str = ""
    if n_clusters >= 2:
        if has_noise:
            mask = np.array(labels) != -1
            if mask.sum() > 1 and len(set(np.array(labels)[mask])) >= 2:
                sil = silhouette_score(X_scaled[mask], np.array(labels)[mask])
                sil_str = f" | Silhouette={sil:.4f}"
        else:
            sil = silhouette_score(X_scaled, labels)
            sil_str = f" | Silhouette={sil:.4f}"

    ax.set_title(f"{title}\n(Klaster={n_clusters}{' + Noise' if has_noise else ''}{sil_str})",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(title="Klaster", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

def save_results(experiment_dir, exp_name, algorithm, params, n_clusters, n_noise, sil, dbi, labels, df_clean):
    """Simpan metrik dan hasil clustering ke folder."""
    # Metrik
    with open(os.path.join(experiment_dir, "metrik.txt"), "w") as f:
        f.write(f"Eksperimen: {exp_name}\nAlgoritma: {algorithm}\nParameter: {params}\n")
        f.write(f"Fitur: {', '.join(FEATURE_NAMES)}\n")
        f.write(f"Observasi: {len(labels)}\nKlaster: {n_clusters}\nNoise: {n_noise}\n")
        if sil is not None: f.write(f"Silhouette: {sil:.4f}\n")
        if dbi is not None: f.write(f"DBI: {dbi:.4f}\n")

    # Hasil CSV
    df_out = df_clean.copy()
    df_out["Klaster"] = labels
    out_cols = ["Provinsi", "Kab/Kota"] + FEATURE_COLS + ["Klaster"]
    df_out[out_cols].to_csv(os.path.join(experiment_dir, "hasil_klaster.csv"), index=False)

def compute_metrics(X_scaled, labels):
    """Hitung Silhouette dan DBI dengan penanganan noise."""
    n_clusters = count_clusters(labels)
    has_noise = any(l == -1 for l in labels)
    sil, dbi = None, None
    if n_clusters >= 2:
        if has_noise:
            mask = np.array(labels) != -1
            if mask.sum() > 1 and len(set(np.array(labels)[mask])) >= 2:
                sil = silhouette_score(X_scaled[mask], np.array(labels)[mask])
                dbi = davies_bouldin_score(X_scaled[mask], np.array(labels)[mask])
        else:
            sil = silhouette_score(X_scaled, labels)
            dbi = davies_bouldin_score(X_scaled, labels)
    return sil, dbi


# ═════════════════════════════════════════════════════════════════════════════
# 1. GAUSSIAN MIXTURE MODEL (GMM)
# ═════════════════════════════════════════════════════════════════════════════

def run_gmm(X_scaled, df_clean, experiment_dir, exp_name, n_components, covariance_type="full", desc=""):
    print(f"\n{'='*60}")
    print(f"  GMM — {desc}")
    print(f"{'='*60}")

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                          random_state=42, n_init=10)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)  # Soft probabilities

    sil, dbi = compute_metrics(X_scaled, labels)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    print(f"  Silhouette : {sil:.4f}")
    print(f"  DBI        : {dbi:.4f}")
    print(f"  BIC        : {bic:.2f}")
    print(f"  AIC        : {aic:.2f}")
    print(f"  Distribusi : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # PCA plot
    plot_pca_result(X_scaled, labels, f"GMM {desc}", os.path.join(experiment_dir, "pca_scatter.png"))

    # Simpan probabilitas soft clustering
    prob_df = df_clean[["Provinsi", "Kab/Kota"]].copy()
    for i in range(n_components):
        prob_df[f"Prob_Klaster_{i}"] = probs[:, i]
    prob_df["Klaster"] = labels
    prob_df.to_csv(os.path.join(experiment_dir, "probabilitas_klaster.csv"), index=False)

    # Simpan hasil
    save_results(experiment_dir, exp_name, "GaussianMixture",
                 f"n_components={n_components}, covariance={covariance_type}",
                 n_components, 0, sil, dbi, labels, df_clean)

    # Print top-5 wilayah paling "ambigu" per klaster
    print(f"\n  >> Wilayah paling AMBIGU (probabilitas terdistribusi):")
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    top_ambig = np.argsort(entropies)[-5:][::-1]
    for idx in top_ambig:
        probs_str = ", ".join([f"K{i}={probs[idx][i]:.2f}" for i in range(n_components)])
        print(f"     {df_clean.iloc[idx]['Kab/Kota']:25s} → [{probs_str}]")

    return labels, sil, dbi, bic, aic


def gmm_find_best_k(X_scaled, df_clean):
    """Cari K optimal untuk GMM berdasarkan BIC dan AIC."""
    print(f"\n{'='*60}")
    print(f"  GMM — AUTO-TUNE (cari K optimal via BIC/AIC)")
    print(f"{'='*60}")

    results = []
    K_range = range(2, 11)
    for k in K_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=10)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        sil, dbi = compute_metrics(X_scaled, labels)
        results.append({
            "K": k, "BIC": gmm.bic(X_scaled), "AIC": gmm.aic(X_scaled),
            "Silhouette": sil, "DBI": dbi,
        })

    res_df = pd.DataFrame(results)
    print(f"\n{res_df.to_string(index=False)}")
    print()

    # Pilih K berdasarkan BIC terendah
    best_bic_k = res_df.loc[res_df["BIC"].idxmin(), "K"]
    best_sil_k = res_df.loc[res_df["Silhouette"].idxmax(), "K"]

    print(f"  🏆 K optimal (BIC terendah)   : K={int(best_bic_k)}")
    print(f"  🏆 K optimal (Silhouette)     : K={int(best_sil_k)}")

    # Simpan
    res_df.to_csv(os.path.join(EXPERIMEN_DIR, "22_gmm_auto", "gmm_bic_aic.csv"), index=False)

    # Plot BIC/AIC
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(list(K_range), res_df["BIC"].values, "o-", color="#2E86AB", linewidth=2, label="BIC")
    ax1.plot(list(K_range), res_df["AIC"].values, "s-", color="#F18F01", linewidth=2, label="AIC")
    ax1.axvline(x=best_bic_k, color="#E8505B", linestyle="--", label=f"Best K (BIC)={int(best_bic_k)}")
    ax1.set_xlabel("Jumlah Komponen (K)", fontsize=12)
    ax1.set_ylabel("Nilai Kriterium", fontsize=12)
    ax1.set_title("GMM — Penentuan K Optimal via BIC & AIC", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMEN_DIR, "22_gmm_auto", "gmm_bic_aic.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return int(best_bic_k), int(best_sil_k)


# ═════════════════════════════════════════════════════════════════════════════
# 2. SPECTRAL CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════

def run_spectral(X_scaled, df_clean, experiment_dir, exp_name, n_clusters, affinity="rbf", desc=""):
    print(f"\n{'='*60}")
    print(f"  SPECTRAL — {desc}")
    print(f"{'='*60}")

    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity=affinity,
        random_state=42, n_init=10,
    )
    labels = spectral.fit_predict(X_scaled)

    sil, dbi = compute_metrics(X_scaled, labels)
    print(f"  Silhouette : {sil:.4f}")
    print(f"  DBI        : {dbi:.4f}")
    print(f"  Distribusi : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    plot_pca_result(X_scaled, labels, f"Spectral {desc}",
                    os.path.join(experiment_dir, "pca_scatter.png"))

    save_results(experiment_dir, exp_name, "SpectralClustering",
                 f"n_clusters={n_clusters}, affinity={affinity}",
                 n_clusters, 0, sil, dbi, labels, df_clean)

    return labels, sil, dbi


# ═════════════════════════════════════════════════════════════════════════════
# 3. MEAN SHIFT
# ═════════════════════════════════════════════════════════════════════════════

def run_mean_shift(X_scaled, df_clean, experiment_dir, exp_name, bandwidth=None):
    print(f"\n{'='*60}")
    print(f"  MEAN SHIFT — bandwidth={bandwidth if bandwidth else 'auto'}")
    print(f"{'='*60}")

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X_scaled, quantile=0.3, n_samples=500)
        print(f"  Bandwidth terestimasi: {bandwidth:.4f}")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True, max_iter=300)
    labels = ms.fit_predict(X_scaled)

    n_clusters = count_clusters(labels)
    sil, dbi = compute_metrics(X_scaled, labels)
    print(f"  Jumlah klaster (auto-detect): {n_clusters}")
    print(f"  Silhouette : {sil:.4f}")
    print(f"  DBI        : {dbi:.4f}")
    print(f"  Distribusi : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    plot_pca_result(X_scaled, labels, f"Mean Shift (bw={bandwidth:.2f})",
                    os.path.join(experiment_dir, "pca_scatter.png"))

    save_results(experiment_dir, exp_name, "MeanShift",
                 f"bandwidth={bandwidth:.4f}, bin_seeding=True",
                 n_clusters, 0, sil, dbi, labels, df_clean)

    # Tampilkan pusat klaster
    cluster_centers = ms.cluster_centers_
    print(f"\n  Pusat klaster (dalam skala asli):")
    scaler = StandardScaler()
    scaler.fit(df_clean[FEATURE_COLS])
    centers_original = scaler.inverse_transform(cluster_centers)
    for i, center in enumerate(centers_original):
        vals = ", ".join([f"{FEATURE_NAMES[j]}={center[j]:.2f}" for j in range(N_FEATURES)])
        print(f"     Klaster {i}: {vals}")

    return labels, sil, dbi, bandwidth


# ═════════════════════════════════════════════════════════════════════════════
# 4. OPTICS
# ═════════════════════════════════════════════════════════════════════════════

def run_optics(X_scaled, df_clean, experiment_dir, exp_name, min_samples=5, xi=0.05, min_cluster_size=0.05, desc=""):
    print(f"\n{'='*60}")
    print(f"  OPTICS — {desc}")
    print(f"{'='*60}")

    optics = OPTICS(
        min_samples=min_samples, xi=xi,
        min_cluster_size=min_cluster_size,
        cluster_method="xi",  # Extract clusters using Xi method
        metric="euclidean",
    )
    optics.fit(X_scaled)
    labels = optics.labels_

    n_clusters = count_clusters(labels)
    n_noise = list(labels).count(-1)
    sil, dbi = compute_metrics(X_scaled, labels)

    print(f"  Jumlah klaster : {n_clusters}")
    print(f"  Noise (outlier): {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"  Silhouette     : {sil:.4f}" if sil is not None else "  Silhouette     : N/A")
    print(f"  DBI            : {dbi:.4f}" if dbi is not None else "  DBI            : N/A")
    print(f"  Distribusi     : {pd.Series(labels).value_counts().sort_index().to_dict()}")

    plot_pca_result(X_scaled, labels, f"OPTICS {desc}",
                    os.path.join(experiment_dir, "pca_scatter.png"))

    save_results(experiment_dir, exp_name, "OPTICS",
                 f"min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size}",
                 n_clusters, n_noise, sil, dbi, labels, df_clean)

    # OPTICS reachability plot
    fig, ax = plt.subplots(figsize=(14, 4))
    space = np.arange(len(X_scaled))
    reachability = optics.reachability_[optics.ordering_]
    ax.plot(space, reachability, "b-", linewidth=1, alpha=0.7)
    ax.set_title("OPTICS — Reachability Plot", fontsize=13, fontweight="bold")
    ax.set_xlabel("Urutan Observasi (berdasarkan ordering_)", fontsize=11)
    ax.set_ylabel("Reachability Distance", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "reachability_plot.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return labels, n_clusters, n_noise, sil, dbi


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  EKSPERIMEN ALGORITMA LANJUTAN")
    print("  GMM + Spectral + Mean Shift + OPTICS")
    print("=" * 72)

    # Load data
    print("\n─── LOAD & PREPARE DATA ───")
    df, df_clean = load_data()
    X_raw, X_scaled, scaler = prepare_data(df_clean)
    print(f"  Data: {X_scaled.shape[0]} observasi, {X_scaled.shape[1]} fitur")

    all_results = []

    # ═══ 1. GMM ═══

    # GMM K=3
    _, sil_g3, dbi_g3, bic3, aic3 = run_gmm(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "20_gmm_k3"),
        "20_gmm_k3", n_components=3,
        desc="K=3, covariance=full",
    )
    all_results.append({"Algoritma": "GMM", "Parameter": "K=3, full", "Klaster": 3,
                        "Silhouette": f"{sil_g3:.4f}", "DBI": f"{dbi_g3:.4f}"})

    # GMM K=4
    _, sil_g4, dbi_g4, bic4, aic4 = run_gmm(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "21_gmm_k4"),
        "21_gmm_k4", n_components=4,
        desc="K=4, covariance=full",
    )
    all_results.append({"Algoritma": "GMM", "Parameter": "K=4, full", "Klaster": 4,
                        "Silhouette": f"{sil_g4:.4f}", "DBI": f"{dbi_g4:.4f}"})

    # GMM Auto-tune (cari K terbaik)
    best_bic_k, best_sil_k = gmm_find_best_k(X_scaled, df_clean)

    # GMM dengan K terbaik dari BIC
    _, sil_gbic, dbi_gbic, bicb, aicb = run_gmm(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "22_gmm_auto"),
        "22_gmm_auto", n_components=best_bic_k,
        desc=f"K={best_bic_k} (auto-BIC)",
    )
    all_results.append({"Algoritma": "GMM", "Parameter": f"K={best_bic_k} (auto-BIC)", "Klaster": best_bic_k,
                        "Silhouette": f"{sil_gbic:.4f}", "DBI": f"{dbi_gbic:.4f}"})

    # ═══ 2. SPECTRAL ═══

    # Spectral K=3
    _, sil_s3, dbi_s3 = run_spectral(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "23_spectral_k3"),
        "23_spectral_k3", n_clusters=3,
        desc="K=3, affinity=rbf",
    )
    all_results.append({"Algoritma": "Spectral", "Parameter": "K=3, rbf", "Klaster": 3,
                        "Silhouette": f"{sil_s3:.4f}", "DBI": f"{dbi_s3:.4f}"})

    # Spectral K=4
    _, sil_s4, dbi_s4 = run_spectral(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "24_spectral_k4"),
        "24_spectral_k4", n_clusters=4,
        desc="K=4, affinity=rbf",
    )
    all_results.append({"Algoritma": "Spectral", "Parameter": "K=4, rbf", "Klaster": 4,
                        "Silhouette": f"{sil_s4:.4f}", "DBI": f"{dbi_s4:.4f}"})

    # ═══ 3. MEAN SHIFT ═══

    _, sil_ms, dbi_ms, bw = run_mean_shift(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "25_mean_shift"),
        "25_mean_shift",
    )
    # Hitung ulang jumlah klaster Mean Shift
    ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True, max_iter=300)
    ms_labels = ms.fit_predict(X_scaled)
    n_ms = count_clusters(ms_labels)
    all_results.append({"Algoritma": "Mean Shift", "Parameter": f"auto (bw={bw:.2f})", "Klaster": n_ms,
                        "Silhouette": f"{sil_ms:.4f}" if sil_ms else "N/A",
                        "DBI": f"{dbi_ms:.4f}" if dbi_ms else "N/A"})

    # ═══ 4. OPTICS ═══

    # OPTICS default
    _, nc_o1, nn_o1, sil_o1, dbi_o1 = run_optics(
        X_scaled, df_clean,
        os.path.join(EXPERIMEN_DIR, "26_optics_default"),
        "26_optics_default",
        min_samples=5, xi=0.05, min_cluster_size=0.05,
        desc="default (min=5, xi=0.05)",
    )
    all_results.append({"Algoritma": "OPTICS", "Parameter": "default (min=5, xi=0.05)", "Klaster": nc_o1,
                        "Silhouette": f"{sil_o1:.4f}" if sil_o1 else "N/A",
                        "DBI": f"{dbi_o1:.4f}" if dbi_o1 else "N/A"})

    # OPTICS tuned (lebih banyak klaster)
    for min_s, xi_val, min_cs, label in [
        (10, 0.1, 0.1, "tuned (min=10, xi=0.1)"),
        (15, 0.3, 0.1, "tuned (min=15, xi=0.3)"),
    ]:
        _, nc, nn, sil_o, dbi_o = run_optics(
            X_scaled, df_clean,
            os.path.join(EXPERIMEN_DIR, "27_optics_tuned"),
            f"27_optics_tuned",
            min_samples=min_s, xi=xi_val, min_cluster_size=min_cs,
            desc=label,
        )
        all_results.append({"Algoritma": "OPTICS", "Parameter": label, "Klaster": nc,
                            "Silhouette": f"{sil_o:.4f}" if sil_o else "N/A",
                            "DBI": f"{dbi_o:.4f}" if dbi_o else "N/A"})


    # ═════════════════════════════════════════════════════════════════════════
    # TABEL PERBANDINGAN LENGKAP (semua algoritma)
    # ═════════════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 72)
    print("  📊 PERBANDINGAN ALGORITMA — LENGKAP (Baru + Sebelumnya)")
    print("=" * 72)

    # Gabungkan dengan hasil sebelumnya
    prev_results = [
        ("K-Means++", "K=3", 3, "0.3291", "0.8900"),
        ("K-Means++", "K=4", 4, "0.4319", "0.7521"),
        ("DBSCAN", "default (eps=0.5, min=5)", 2, "0.4303", "0.4529"),
        ("DBSCAN", "tuned (eps=1.5, min=3)", 2, "0.7793", "0.2164"),
        ("Agglomerative (ward)", "K=3", 3, "0.4724", "0.8745"),
        ("Agglomerative (ward)", "K=4", 4, "0.4526", "0.6929"),
        ("Agglomerative (complete)", "K=4", 4, "0.4542", "0.6113"),
    ]

    for alg, param, k, sil, dbi in prev_results:
        all_results.append({"Algoritma": alg, "Parameter": param, "Klaster": k,
                            "Silhouette": sil, "DBI": dbi})

    result_df = pd.DataFrame(all_results)
    result_df["_sil_num"] = pd.to_numeric(result_df["Silhouette"], errors="coerce").fillna(0)
    result_df = result_df.sort_values("_sil_num", ascending=False).drop(columns=["_sil_num"]).reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "Rank"

    print(f"\n{result_df.to_string()}")
    print()

    # Simpan
    result_df.to_csv(os.path.join(EXPERIMEN_DIR, "perbandingan_algoritma_lengkap.csv"))
    print(f"  → eksperimen/perbandingan_algoritma_lengkap.csv")

    # ─── REKOMENDASI ───
    print("\n" + "=" * 72)
    print("  💡 REKOMENDASI FINAL — SEMUA ALGORITMA")
    print("=" * 72)

    valid = [(i, r["Algoritma"], r["Parameter"], r["Klaster"], r["Silhouette"], r["DBI"])
             for i, r in result_df.iterrows() if r["Silhouette"] != "N/A"]

    print(f"\n  🏆 TOP 5 KONFIGURASI TERBAIK (berdasarkan Silhouette Score):")
    for rank, alg, param, k, sil, dbi in valid[:5]:
        print(f"    #{rank} — {alg:30s} | {param:30s} | K={k} | Sil={sil} | DBI={dbi}")

    # Analisis khusus untuk setiap algoritma baru
    print(f"\n")
    print(f"  🔍 ANALISIS PER ALGORITMA BARU:")
    print(f"  {'─'*60}")
    print(f"  GMM K=3        : Sil={sil_g3:.4f}, DBI={dbi_g3:.4f} — Soft clustering")
    print(f"  GMM K=4        : Sil={sil_g4:.4f}, DBI={dbi_g4:.4f} — Soft clustering")
    print(f"  GMM auto-BIC   : K={best_bic_k}, Sil={sil_gbic:.4f} — Optimal via BIC")
    print(f"  Spectral K=3   : Sil={sil_s3:.4f}, DBI={dbi_s3:.4f} — Graph-based")
    print(f"  Spectral K=4   : Sil={sil_s4:.4f}, DBI={dbi_s4:.4f} — Graph-based")
    print(f"  Mean Shift     : K={n_ms}, Sil={sil_ms:.4f} — Auto-detect")
    print(f"  OPTICS default : K={nc_o1}, Sil={sil_o1 if sil_o1 else 'N/A':>8} — Density-based")

    # Kesimpulan
    print(f"\n")
    print(f"  📌 KESIMPULAN:")
    print(f"  {'─'*60}")
    print(f"  1. Agglomerative Ward K=3 (Sil=0.4724) masih terbaik untuk interpretasi + metrik")
    print(f"  2. GMM K=3 (Sil={sil_g3:.4f}) kompetitif dengan Agglomerative — plus bisa lihat probabilitas!")
    print(f"  3. Mean Shift detect K={n_ms} secara otomatis — menarik untuk eksplorasi")
    print(f"  4. Spectral dan OPTICS kurang optimal untuk data 3 fitur ini")

    print("\n" + "=" * 72)
    print("  EKSPERIMEN SELESAI! 🎉")
    print("=" * 72)


if __name__ == "__main__":
    main()
