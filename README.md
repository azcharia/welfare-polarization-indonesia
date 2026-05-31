# Welfare Polarization and Regional Economic Saturation Analysis in Indonesia

An unsupervised machine learning and data science project designed to analyze spatial economic polarization (the Growth Paradox / Regional Economic Saturation) across 514 districts and cities (Kabupaten/Kota) in Indonesia. 

The project segments regions based on three primary indicators from BPS (Statistics Indonesia):
1. **GRDP (PDRB)**: Gross Regional Domestic Product at Constant Prices.
2. **TPT**: Open Unemployment Rate (Tingkat Pengangguran Terbuka).
3. **P0**: Percentage of Poor Population (Persentase Penduduk Miskin).

---

## 🚀 Key Features

* **Multi-Algorithm Clustering**: Evaluation and implementation of various clustering models:
  * K-Means++ (Recommended $K=4$ and Baseline $K=3$)
  * Agglomerative Hierarchical Clustering (Ward linkage, complete linkage)
  * Spectral Clustering
  * Gaussian Mixture Models (GMM)
  * DBSCAN & OPTICS (Density-based)
  * Mean Shift
* **Dimensionality Reduction**: Principal Component Analysis (PCA) for 2D/3D visual cluster mapping.
* **Comprehensive Metrics**: Model validation using:
  * Silhouette Score
  * Davies-Bouldin Index (DBI)
  * Calinski-Harabasz Index
  * Dunn Index (custom implementation)
  * Gap Statistic (custom implementation vs. uniform reference)
* **Interactive Dashboard**: Streamlit-based web application with:
  * Dynamic filters by Province and Region.
  * Interactive sliders for GRDP, TPT, and P0 ranges.
  * Visualizations like PCA Scatter Plots, Cluster Profiling Radar Charts, Bar Charts, and Typology Distributions.
* **Automated Report Generation**: Node.js compiler script utilizing `docx` library to dynamically generate a comprehensive Word report (`.docx`) summarizing all 28 experiments and metrics.

---

## 📁 Project Structure

```text
├── dashboard/
│   ├── app.py             # Streamlit main entry point
│   ├── requirements.txt   # Python dependencies for the dashboard
│   └── src/               # Modular scripts
│       ├── data.py        # Data loading, preprocessing, and caching
│       ├── clustering.py  # Model executions, PCA, and metrics
│       └── viz.py         # Plotly interactive visualizations
├── dataset/
│   └── Klasifikasi Tingkat Kemiskinan di Indonesia.csv  # BPS dataset
├── eksperimen/            # 28 distinct clustering configurations
│   ├── run_all.py         # Main script executing feature combinations
│   ├── run_algoritma_lain.py      # Baseline comparisons (DBSCAN, Agglomerative)
│   ├── run_algoritma_lanjutan.py  # Advanced algorithms (GMM, Spectral, OPTICS)
│   ├── run_metrik_lengkap.py     # Evaluation across 5 validation metrics
│   └── ... (numbered experiment output directories)
├── dokumen/               # Project references and PDFs
├── polarisasi_kesejahteraan_kmeans_pca.py # Core baseline pipeline script
├── generate_dokumentasi.js                # Document compiler (Node.js)
├── package.json           # Node.js dependencies for report generator
├── .gitignore             # Configured git ignore patterns
└── README.md              # This file
```

---

## 🛠️ Installation & Usage

### 1. Prerequisites
Ensure you have the following installed:
* Python 3.9+
* Node.js 18+ (optional, only needed for document generation)

### 2. Python Setup & Running the Dashboard
Clone the repository, set up a virtual environment, and install python dependencies:

```bash
# Setup Virtual Environment (Optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r dashboard/requirements.txt

# Run the Streamlit Dashboard
streamlit run dashboard/app.py
```

### 3. Running the Base Pipeline
To run the primary clustering pipeline and output the static charts (`Evaluasi_K_Optimal.png` and `Visualisasi_PCA_Klaster_Polarisasi.png`) and dataset `Hasil_Klaster_Polarisasi_Kesejahteraan.csv`:

```bash
python polarisasi_kesejahteraan_kmeans_pca.py
```

### 4. Running Experiments
To replicate the 28 experimental configurations and benchmark evaluations:

```bash
cd eksperimen
python run_all.py
python run_algoritma_lain.py
python run_algoritma_lanjutan.py
python run_metrik_lengkap.py
```

### 5. Generating the Experiment Report (.docx)
To compile the comprehensive Word document report based on the experiments:

```bash
# In the root folder
npm install
node generate_dokumentasi.js
```

---

## 📊 Typology Classifications
The model segments the 514 districts/cities into distinct profiles (referred to as **Tipologi Polarisasi**):
1. **Episentrum Ekonomi Jenuh / Rentan**: Regions with extremely high GRDP but paradoxical high unemployment rate (TPT).
2. **Kerentanan Struktural**: Low-income regions characterized by high poverty rates (P0).
3. **Wilayah Transisi / Berkembang**: Emerging or transition economies with balanced mid-tier indicators.

---

## 📄 License & Credits
* **Data Sources**: BPS (Badan Pusat Statistik) Indonesia.
* **Contributors**: Naufal A, Rahma, Sita.
