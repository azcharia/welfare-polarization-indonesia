"""
Evaluasi Model Clustering dengan 5 Metrik Lengkap
==================================================
Metrik:
1. Silhouette Score        (↑ baik) — sklearn
2. Davies-Bouldin Index    (↓ baik) — sklearn
3. Calinski-Harabasz Index (↑ baik) — sklearn
4. Dunn Index              (↑ baik) — custom (sensitive to outliers)
5. Gap Statistic           (↑ baik) — custom (compare vs random)

Penjelasan Dunn Index:
  Dunn = min_intercluster_distance / max_intracluster_distance
  - SANGAT sensitif ke outlier — 1 titik outlier bisa turunkan skor drastis
  - Makin tinggi = cluster makin kompak & terpisah

Penjelasan Gap Statistic:
  Gap(k) = E*[log(W_k)] - log(W_k)
  - Bandingkan inertia data asli vs data random
  - Makin tinggi = cluster makin bermakna (tidak random)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform, cdist
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(PROJECT_DIR, "dataset", "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")
df = pd.read_csv(FILE_PATH)

FEATURE_COLS = [
    'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)',
    'Tingkat Pengangguran Terbuka',
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)',
]

FEATURE_NAMES = ["PDRB", "TPT", "P0"]
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
X_raw = df[FEATURE_COLS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

N = len(X_scaled)
print(f"Data: {N} kabupaten/kota, {len(FEATURE_COLS)} fitur")
print(f"Fitur: {', '.join(FEATURE_NAMES)}")
print()

# ============================================================
# 2. IMPLEMENTASI METRIK TAMBAHAN
# ============================================================

def dunn_index(X, labels):
    """
    Dunn Index = min_intercluster_distance / max_intracluster_distance
    
    - min_intercluster_distance: jarak minimum antara 2 titik dari cluster berbeda
    - max_intracluster_distance: diameter maksimum dalam satu cluster
    
    ↑ HIGHER is better.
    SANGAT sensitif ke outlier — 1 titik outlier bisa bikin nilai rendah.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # skip noise (-1)
    
    if len(unique_labels) < 2:
        return None
    
    # Distance matrix
    dist_matrix = squareform(pdist(X, metric='euclidean'))
    
    # Intra-cluster: max diameter per cluster
    max_diameter = 0
    for label in unique_labels:
        mask = labels == label
        cluster_points = np.where(mask)[0]
        if len(cluster_points) > 1:
            # Diameter = max distance between any 2 points in same cluster
            cluster_dist = dist_matrix[np.ix_(cluster_points, cluster_points)]
            diameter = np.max(cluster_dist)
            if diameter > max_diameter:
                max_diameter = diameter
        # Kalau cuma 1 titik, diameter = 0 (tidak pengaruh)
    
    if max_diameter == 0:
        return None
    
    # Inter-cluster: min distance between any 2 points from different clusters
    min_separation = np.inf
    for i, label_i in enumerate(unique_labels):
        mask_i = labels == label_i
        points_i = np.where(mask_i)[0]
        for label_j in unique_labels[i+1:]:
            mask_j = labels == label_j
            points_j = np.where(mask_j)[0]
            # Minimum distance between points in cluster i and cluster j
            distances = dist_matrix[np.ix_(points_i, points_j)]
            min_dist = np.min(distances)
            if min_dist < min_separation:
                min_separation = min_dist
    
    return min_separation / max_diameter


def gap_statistic(X, labels, n_refs=10, random_state=42):
    """
    Gap Statistic — bandingkan inertia data asli vs data random.
    
    Gap(k) = (1/B) * sum(log(W_kb*)) - log(W_k)
    
    ↑ HIGHER gap = cluster lebih bermakna (tidak random)
    
    Parameter:
    - n_refs: jumlah dataset random untuk referensi
    """
    rng = np.random.RandomState(random_state)
    n_clusters = len(np.unique(labels))
    
    # Inertia untuk data asli
    inertia_original = _compute_inertia(X, labels, n_clusters)
    
    if inertia_original is None or inertia_original == 0:
        return None
    
    log_w_original = np.log(inertia_original)
    
    # Inertia untuk data random
    # Sample uniform dalam bounding box data asli
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    log_w_refs = []
    for _ in range(n_refs):
        X_random = rng.uniform(mins, maxs, size=X.shape)
        
        # K-Means dengan K yang sama
        km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5, random_state=rng.randint(1000))
        random_labels = km.fit_predict(X_random)
        
        inertia_random = _compute_inertia(X_random, random_labels, n_clusters)
        if inertia_random and inertia_random > 0:
            log_w_refs.append(np.log(inertia_random))
    
    if len(log_w_refs) == 0:
        return None
    
    gap = np.mean(log_w_refs) - log_w_original
    
    # Standard deviation (untuk confidence)
    sd = np.std(log_w_refs)
    sdk = sd * np.sqrt(1 + 1.0 / n_refs)
    
    return gap, sdk


def _compute_inertia(X, labels, n_clusters):
    """Hitung inertia (WCSS) untuk label tertentu."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    
    if len(unique_labels) < 2:
        return None
    
    inertia = 0
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
    
    return inertia


# ============================================================
# 3. DEFINISI MODEL-MODEL TERBAIK
# ============================================================

models = [
    # (name, fit_predict_function, requires_scaled, n_clusters_display)
    
    # === K-Means ===
    ("K-Means K=3", 
     lambda: KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42).fit_predict(X_scaled),
     None),
    
    ("K-Means K=4", 
     lambda: KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42).fit_predict(X_scaled),
     None),
    
    # === Agglomerative ===
    ("Agglomerative Ward K=3", 
     lambda: AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_scaled),
     None),
    
    ("Agglomerative Ward K=4", 
     lambda: AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(X_scaled),
     None),
    
    ("Agglomerative Complete K=4", 
     lambda: AgglomerativeClustering(n_clusters=4, linkage='complete').fit_predict(X_scaled),
     None),
    
    # === Spectral ===
    ("Spectral K=3", 
     lambda: SpectralClustering(n_clusters=3, affinity='rbf', random_state=42, n_init=10).fit_predict(X_scaled),
     None),
    
    ("Spectral K=4", 
     lambda: SpectralClustering(n_clusters=4, affinity='rbf', random_state=42, n_init=10).fit_predict(X_scaled),
     None),
    
    # === Mean Shift ===
    ("Mean Shift (auto)", 
     lambda: _mean_shift_fit(X_scaled),
     None),
    
    # === DBSCAN ===
    ("DBSCAN default (eps=0.5)", 
     lambda: DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled),
     None),
    
    ("DBSCAN tuned (eps=1.5)", 
     lambda: DBSCAN(eps=1.5, min_samples=3).fit_predict(X_scaled),
     None),
    
    # === GMM ===
    ("GMM K=3", 
     lambda: GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit_predict(X_scaled),
     None),
    
    ("GMM K=4", 
     lambda: GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit_predict(X_scaled),
     None),
]


def _mean_shift_fit(X):
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500, random_state=42)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True, max_iter=300)
    return ms.fit_predict(X)


# ============================================================
# 4. EVALUASI SEMUA MODEL
# ============================================================

results = []

for name, fit_fn, _ in models:
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    
    try:
        labels = fit_fn()
        n_clusters = len(np.unique(labels))
        n_noise = np.sum(labels == -1)
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        n_valid = len(valid_labels)
        
        # --- Silhouette Score ---
        if n_valid >= 2:
            mask_valid = labels >= 0
            sil = silhouette_score(X_scaled[mask_valid], labels[mask_valid])
        else:
            sil = None
        
        # --- Davies-Bouldin Index ---
        if n_valid >= 2:
            dbi = davies_bouldin_score(X_scaled[mask_valid], labels[mask_valid])
        else:
            dbi = None
        
        # --- Calinski-Harabasz Index ---
        if n_valid >= 2:
            ch = calinski_harabasz_score(X_scaled[mask_valid], labels[mask_valid])
        else:
            ch = None
        
        # --- Dunn Index ---
        dunn = dunn_index(X_scaled, labels)
        
        # --- Gap Statistic ---
        gap_result = gap_statistic(X_scaled, labels, n_refs=10)
        if gap_result:
            gap, gap_sd = gap_result
        else:
            gap, gap_sd = None, None
        
        # --- Summary ---
        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
        dbi_str = f"{dbi:.4f}" if dbi is not None else "N/A"
        ch_str = f"{ch:.2f}" if ch is not None else "N/A"
        dunn_str = f"{dunn:.4f}" if dunn is not None else "N/A"
        gap_str = f"{gap:.4f}" if gap is not None else "N/A"
        gap_sd_str = f"±{gap_sd:.4f}" if gap_sd is not None else ""
        
        print(f"  Klaster         : {n_clusters} (valid: {n_valid}, noise: {n_noise})")
        print(f"  Distribusi      : {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"  1. Silhouette    : {sil_str}")
        print(f"  2. DBI           : {dbi_str}")
        print(f"  3. Calinski-Harabasz : {ch_str}")
        print(f"  4. Dunn Index    : {dunn_str}")
        print(f"  5. Gap Statistic : {gap_str} {gap_sd_str}")
        
        results.append({
            "Model": name,
            "Klaster": n_clusters,
            "Valid": n_valid,
            "Noise": n_noise,
            "Silhouette↑": sil,
            "DBI↓": dbi,
            "CH↑": ch,
            "Dunn↑": dunn,
            "Gap↑": gap,
        })
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        print(f"  Silhouette    : N/A")
        print(f"  DBI           : N/A")
        print(f"  CH Index      : N/A")
        print(f"  Dunn Index    : N/A")
        print(f"  Gap Statistic : N/A")


# ============================================================
# 5. TABEL PERBANDINGAN FINAL
# ============================================================

print("\n\n")
print("="*100)
print("📊  TABEL PERBANDINGAN — 5 METRIK LENGKAP")
print("="*100)
print(f"{'No':<3} {'Model':<30} {'K':<3} {'Silhouette↑':<12} {'DBI↓':<10} {'CH↑':<12} {'Dunn↑':<10} {'Gap↑':<10}")
print("-"*100)

# Sort by Silhouette (best first)
sorted_results = sorted(results, key=lambda r: (r["Silhouette↑"] if r["Silhouette↑"] is not None else -1), reverse=True)

for i, r in enumerate(sorted_results, 1):
    sil = f"{r['Silhouette↑']:.4f}" if r['Silhouette↑'] is not None else "N/A"
    dbi = f"{r['DBI↓']:.4f}" if r['DBI↓'] is not None else "N/A"
    ch = f"{r['CH↑']:.2f}" if r['CH↑'] is not None else "N/A"
    dunn = f"{r['Dunn↑']:.4f}" if r['Dunn↑'] is not None else "N/A"
    gap = f"{r['Gap↑']:.4f}" if r['Gap↑'] is not None else "N/A"
    
    print(f"{i:<3} {r['Model']:<30} {r['Klaster']:<3} {sil:<12} {dbi:<10} {ch:<12} {dunn:<10} {gap:<10}")

print("="*100)


# ============================================================
# 6. VISUALISASI PERBANDINGAN
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    model_names = [r['Model'] for r in sorted_results]
    sil_vals = [r['Silhouette↑'] if r['Silhouette↑'] is not None else 0 for r in sorted_results]
    dbi_vals = [r['DBI↓'] if r['DBI↓'] is not None else 0 for r in sorted_results]
    ch_vals = [r['CH↑'] if r['CH↑'] is not None else 0 for r in sorted_results]
    dunn_vals = [r['Dunn↑'] if r['Dunn↑'] is not None else 0 for r in sorted_results]
    gap_vals = [r['Gap↑'] if r['Gap↑'] is not None else 0 for r in sorted_results]
    
    # Normalize untuk radar chart: bagi dengan max value
    def normalize(vals):
        max_v = max(vals) if max(vals) > 0 else 1
        return [v / max_v for v in vals]
    
    sil_norm = normalize(sil_vals)
    ch_norm = normalize(ch_vals)
    dunn_norm = normalize(dunn_vals)
    gap_norm = normalize(gap_vals)
    dbi_norm = [1 - (v / max(dbi_vals)) if max(dbi_vals) > 0 else 0 for v in dbi_vals]  # DBI dibalik
    
    # --- Bar Chart: Semua Metrik ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Perbandingan 5 Metrik Clustering — Semua Model', fontsize=14, fontweight='bold')
    
    # Gunakan nama pendek untuk label
    short_names = [n.replace('Agglomerative ', 'Agg.').replace(' (auto)', '') for n in model_names]
    
    metrics_data = [
        (axes[0,0], sil_vals, 'Silhouette Score ↑', 'skyblue', 'higher is better'),
        (axes[0,1], dbi_vals, "Davies-Bouldin Index ↓", 'salmon', 'lower is better'),
        (axes[1,0], ch_vals, 'Calinski-Harabasz ↑', 'lightgreen', 'higher is better'),
        (axes[1,1], dunn_vals, 'Dunn Index ↑', 'plum', 'higher is better (sensitive)'),
        (axes[2,0], gap_vals, 'Gap Statistic ↑', 'gold', 'higher is better'),
    ]
    
    for ax, vals, title, color, note in metrics_data:
        bars = ax.bar(range(len(vals)), vals, color=color, edgecolor='gray', alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        # Add note
        ax.text(0.5, -0.15, note, transform=ax.transAxes, ha='center', fontsize=8, color='gray')
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(vals)*0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    axes[2,1].axis('off')
    
    plt.tight_layout()
    plt.savefig("eksperimen/28_evaluasi_metrik_lengkap/perbandingan_5_metrik.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Grafik perbandingan tersimpan: perbandingan_5_metrik.png")
    
    # --- Radar Chart: 5 Model Teratas ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle('Radar Chart — 5 Model Terbaik', fontsize=14, fontweight='bold')
    
    # Pilih 5 model teratas
    top5 = sorted_results[:5]
    top5_names = [n.replace('Agglomerative ', 'Agg.').replace(' (auto)', '') for n in [r['Model'] for r in top5]]
    
    categories = ['Silhouette', 'CH Index', 'Dunn Index', 'Gap Stat.', 'DBI (inv)']
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, r in enumerate(top5):
        values = [
            r['Silhouette↑'] if r['Silhouette↑'] is not None else 0,
            r['CH↑'] / max(ch_vals) if r['CH↑'] is not None else 0,
            r['Dunn↑'] / max(dunn_vals) if r['Dunn↑'] is not None else 0,
            r['Gap↑'] / max(gap_vals) if r['Gap↑'] is not None else 0,
            1 - (r['DBI↓'] / max(dbi_vals)) if r['DBI↓'] is not None else 0,
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f"{idx+1}. {top5_names[idx]}", color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("eksperimen/28_evaluasi_metrik_lengkap/radar_5_model_terbaik.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Radar chart tersimpan: radar_5_model_terbaik.png")
    
except Exception as e:
    print(f"\n⚠️  Visualisasi skip: {e}")


# ============================================================
# 7. SIMPAN HASIL KE CSV
# ============================================================
results_df = pd.DataFrame(sorted_results)
results_df = results_df.rename(columns={
    "Silhouette↑": "Silhouette",
    "DBI↓": "DBI",
    "CH↑": "Calinski_Harabasz",
    "Dunn↑": "Dunn_Index",
    "Gap↑": "Gap_Statistic",
})
results_df.to_csv("eksperimen/28_evaluasi_metrik_lengkap/perbandingan_5_metrik.csv", index=False)
print(f"\n✅ Hasil tersimpan: perbandingan_5_metrik.csv")

# Print ranked berdasarkan setiap metrik
print("\n\n")
print("="*100)
print("🏆  RANKING PER METRIK")
print("="*100)

for metric_name, metric_key, higher_better in [
    ("Silhouette Score ↑", "Silhouette↑", True),
    ("DBI ↓", "DBI↓", False),
    ("Calinski-Harabasz ↑", "CH↑", True),
    ("Dunn Index ↑", "Dunn↑", True),
    ("Gap Statistic ↑", "Gap↑", True),
]:
    valid_results = [r for r in results if r[metric_key] is not None]
    sorted_by_metric = sorted(valid_results, key=lambda r: r[metric_key], reverse=higher_better)
    
    print(f"\n📊 {metric_name}")
    print("-" * 70)
    for rank, r in enumerate(sorted_by_metric[:5], 1):
        val = r[metric_key]
        if val is not None:
            print(f"  #{rank}  {r['Model']:<35} → {val:.4f}")
