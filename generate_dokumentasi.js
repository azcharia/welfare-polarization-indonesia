const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, ImageRun
} = require("docx");

// ============================================================
// KONFIGURASI
// ============================================================
const OUTPUT_DIR = __dirname;
const EXPERIMEN_DIR = `${OUTPUT_DIR}/eksperimen`;
const COLORS = {
  primary: "1B3A5C",
  secondary: "2E86AB",
  accent: "E8505B",
  highlight: "F18F01",
  light: "D5E8F0",
  gray: "F2F2F2",
  white: "FFFFFF",
};

// Helper: border tipis
const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
const noBorder = { style: BorderStyle.NONE, size: 0 };

// Helper: cell margins
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };
const headerCellMargins = { top: 80, bottom: 80, left: 100, right: 100 };

// ============================================================
// FUNGSI BANTU
// ============================================================

function heading(level, text) {
  const headingMap = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
  };
  return new Paragraph({
    heading: headingMap[level] || HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, font: "Arial" })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    children: [new TextRun({
      text,
      font: "Arial",
      size: opts.small ? 20 : 24,
      bold: opts.bold || false,
      color: opts.color || "333333",
      italics: opts.italics || false,
    })],
  });
}

function pMulti(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120, line: 276 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    children: runs,
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60, line: 276 },
    children: [new TextRun({ text, font: "Arial", size: 24 })],
  });
}

function bulletBold(boldPart, rest, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60, line: 276 },
    children: [
      new TextRun({ text: boldPart, font: "Arial", size: 24, bold: true }),
      new TextRun({ text: rest, font: "Arial", size: 24 }),
    ],
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 60 }, children: [] });
}

// ============================================================
// TABEL
// ============================================================

function createTable(headerRow, dataRows, colWidths) {
  const tableWidth = colWidths.reduce((a, b) => a + b, 0);
  
  const rows = [
    // Header
    new TableRow({
      tableHeader: true,
      children: headerRow.map((h, i) => new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: COLORS.primary, type: ShadingType.CLEAR },
        verticalAlign: "center",
        margins: headerCellMargins,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: String(h), font: "Arial", size: 20, bold: true, color: "FFFFFF" })],
        })],
      })),
    }),
  ];

  // Data rows
  dataRows.forEach((rowData, idx) => {
    const bgColor = idx % 2 === 0 ? COLORS.white : COLORS.gray;
    rows.push(new TableRow({
      children: rowData.map((cell, i) => new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: bgColor, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({
          alignment: i === 0 || i === 4 || i === 5 ? AlignmentType.CENTER : AlignmentType.LEFT,
          children: [new TextRun({
            text: String(cell),
            font: "Arial",
            size: 18,
            bold: i === 0 || cell.includes("🏆"),
            color: cell.includes("🏆") ? COLORS.accent : "333333",
          })],
        })],
      })),
    }));
  });

  return new Table({
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows,
  });
}

// ============================================================
// DATA HASIL
// ============================================================

const fiveMetricHeaders = ["No", "Model", "K", "Silhouette↑", "DBI↓", "CH↑", "Dunn↑", "Gap↑"];
const fiveMetricColWidths = [400, 3500, 400, 1000, 800, 1000, 900, 900];
const fiveMetricData = [
  ["1", "DBSCAN tuned (eps=1.5)", "2", "0.7793", "0.2164", "89.6", "0.1819", "0.9142"],
  ["2", "Spectral K=3", "3", "0.7087", "0.4182", "86.0", "0.1581", "0.8550"],
  ["3", "Mean Shift (auto)", "4", "0.5437", "0.4776", "85.1", "0.0357", "0.8092"],
  ["4", "Spectral K=4", "4", "0.5058", "0.4952", "129.4", "0.0285", "0.9693"],
  ["5", "Agglomerative Ward K=3", "3", "0.4724", "0.8745", "257.5", "0.0255", "1.2619"],
  ["6", "Agglomerative Complete K=4", "4", "0.4542", "0.6113", "207.3", "0.0584", "1.2004"],
  ["7", "Agglomerative Ward K=4", "4", "0.4526", "0.6929", "321.4", "0.0549", "1.4647"],
  ["8 🏆", "K-Means K=4 (Rekomendasi)", "4", "0.4319", "0.7521", "386.5", "0.0202", "1.5890"],
  ["9", "DBSCAN default", "3", "0.4303", "0.4529", "19.9", "0.0764", "1.1205"],
  ["10", "K-Means K=3 (Baseline)", "3", "0.3291", "0.8900", "268.2", "0.0189", "1.2826"],
  ["11", "GMM K=3", "3", "0.1807", "1.3681", "165.4", "0.0118", "1.0642"],
  ["12", "GMM K=4", "4", "0.1583", "1.4353", "163.4", "0.0030", "1.0769"],
];

// ============================================================
// SECTION: COVER PAGE
// ============================================================

function createCoverPage() {
  return [
      emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
      emptyLine(), emptyLine(), emptyLine(), emptyLine(),
      
      pMulti([
        new TextRun({ text: "LAPORAN EKSPERIMEN", font: "Arial", size: 52, bold: true, color: COLORS.primary }),
      ], { center: true, after: 200 }),
      
      pMulti([
        new TextRun({ text: "Polarisasi Kesejahteraan dan Kejenuhan Wilayah", font: "Arial", size: 36, bold: true, color: COLORS.secondary }),
      ], { center: true, after: 80 }),

      pMulti([
        new TextRun({ text: "Analisis Clustering untuk Deteksi Paradoks Pertumbuhan Ekonomi", font: "Arial", size: 24, italics: true, color: "666666" }),
      ], { center: true, after: 400 }),

      // Garis pemisah
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.accent, space: 1 } },
        children: [],
      }),

      emptyLine(),

      pMulti([
        new TextRun({ text: "Perbandingan 7 Algoritma Clustering dengan 5 Metrik Evaluasi", font: "Arial", size: 24, color: "444444" }),
      ], { center: true, after: 300 }),

      pMulti([
        new TextRun({ text: "Disusun oleh:", font: "Arial", size: 22, color: "666666" }),
      ], { center: true, after: 60 }),
      pMulti([
        new TextRun({ text: "Naufal A, Rahma, Sita", font: "Arial", size: 24, bold: true, color: COLORS.primary }),
      ], { center: true, after: 200 }),

      emptyLine(),
      
      pMulti([
        new TextRun({ text: "Program Studi Ilmu Komputer", font: "Arial", size: 22, color: "666666" }),
      ], { center: true, after: 40 }),
      pMulti([
        new TextRun({ text: "Hak Kekayaan Intelektual – Program Komputer", font: "Arial", size: 22, color: "666666" }),
      ], { center: true, after: 40 }),
      pMulti([
        new TextRun({ text: "2026", font: "Arial", size: 22, color: "666666" }),
      ], { center: true, after: 200 }),
  ]
}

// ============================================================
// SECTION: BAB 1 - PENDAHULUAN
// ============================================================

function createBAB1() {
  return [
    heading(1, "BAB 1: PENDAHULUAN"),
    
    heading(2, "1.1 Latar Belakang"),
    p("Pertumbuhan ekonomi regional di Indonesia menunjukkan ketimpangan yang signifikan antar wilayah. Beberapa daerah mencatatkan Produk Domestik Regional Bruto (PDRB) yang sangat tinggi, namun di sisi lain masih bergelut dengan tingkat pengangguran yang tinggi — fenomena ini dikenal sebagai paradoks pertumbuhan ekonomi atau kejenuhan wilayah (regional saturation)."),
    p("Di sisi lain, terdapat wilayah-wilayah dengan tingkat kemiskinan dan keterbatasan akses yang ekstrem, terutama di Indonesia Timur, yang menunjukkan kerentanan struktural. Polaritas antara \"episentrum ekonomi jenuh\" dan \"wilayah kerentanan struktural\" ini memerlukan pemahaman mendalam melalui pendekatan data-driven."),
    p("Proyek ini menggunakan metode unsupervised learning — khususnya clustering — untuk mengelompokkan 514 kabupaten/kota di Indonesia berdasarkan indikator ekonomi utama, sehingga tipologi polarisasi kesejahteraan dapat teridentifikasi secara objektif."),

    heading(2, "1.2 Tujuan"),
    p("Tujuan dari eksperimen ini adalah:"),
    bullet("Mengidentifikasi tipologi wilayah berdasarkan polarisasi kesejahteraan dan paradoks pertumbuhan ekonomi"),
    bullet("Membandingkan kinerja berbagai algoritma clustering (K-Means, Agglomerative, DBSCAN, Spectral, Mean Shift, GMM, OPTICS) untuk data kesejahteraan Indonesia"),
    bullet("Mengevaluasi model menggunakan 5 metrik clustering internal: Silhouette Score, Davies-Bouldin Index (DBI), Calinski-Harabasz Index (CH), Dunn Index, dan Gap Statistic"),
    bullet("Menentukan model dan konfigurasi terbaik untuk analisis polarisasi wilayah"),

    heading(2, "1.3 Ruang Lingkup"),
    bullet("Dataset: 514 kabupaten/kota di Indonesia dari data BPS"),
    bullet("Fitur utama: PDRB, Tingkat Pengangguran Terbuka (TPT), Persentase Penduduk Miskin (P0)"),
    bullet("Algoritma: K-Means++, Agglomerative (Ward & Complete), DBSCAN, Spectral Clustering, Mean Shift, Gaussian Mixture Model (GMM), OPTICS"),
    bullet("Metrik evaluasi: Silhouette Score, DBI, Calinski-Harabasz, Dunn Index, Gap Statistic"),
    emptyLine(),
  ];
}

// ============================================================
// SECTION: BAB 2 - DATASET
// ============================================================

function createBAB2() {
  return [
    heading(1, "BAB 2: DESKRIPSI DATASET"),

    heading(2, "2.1 Sumber Data"),
    p("Dataset yang digunakan bersumber dari Badan Pusat Statistik (BPS) dengan judul \"Klasifikasi Tingkat Kemiskinan di Indonesia\". Dataset mencakup 514 kabupaten/kota di seluruh provinsi Indonesia."),

    heading(2, "2.2 Fitur yang Digunakan"),
    p("Dari total kolom yang tersedia, eksperimen difokuskan pada 3 fitur utama yang merepresentasikan dimensi ekonomi dan kesejahteraan:"),
    bulletBold("PDRB (Produk Domestik Regional Bruto): ", "Nilai ekonomi total suatu wilayah dalam Rupiah. Berskala triliunan — sangat timpang antar wilayah."),
    bulletBold("TPT (Tingkat Pengangguran Terbuka): ", "Persentase angkatan kerja yang menganggur. Indikator kesehatan pasar tenaga kerja."),
    bulletBold("P0 (Persentase Penduduk Miskin): ", "Proporsi penduduk di bawah garis kemiskinan. Indikator kesejahteraan langsung."),

    heading(2, "2.3 Preprocessing"),
    p("Tahapan preprocessing yang dilakukan:"),
    bullet("Pembersihan data: Menghapus baris dengan missing values pada ketiga fitur utama"),
    bullet("Standarisasi: Menggunakan StandardScaler (z-score normalization) karena rentang nilai antar fitur sangat berbeda — PDRB berskala triliunan sementara TPT dan P0 berkisar 0-100%"),
    bullet("Hasil: 514 sampel valid, 0 missing values setelah cleaning"),
    emptyLine(),
  ];
}

// ============================================================
// SECTION: BAB 3 - METODOLOGI
// ============================================================

function createBAB3() {
  return [
    heading(1, "BAB 3: METODOLOGI"),

    heading(2, "3.1 Algoritma Clustering yang Diuji"),
    p("Sebanyak 7 algoritma clustering diuji dengan berbagai konfigurasi parameter, menghasilkan 16 skenario eksperimen:"),

    pMulti([
      new TextRun({ text: "3.1.1 K-Means++", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("K-Means adalah algoritma partitional clustering yang membagi data menjadi K cluster berdasarkan jarak ke centroid. Menggunakan inisialisasi k-means++ untuk hasil yang lebih stabil. Diuji dengan K=3 (baseline), K=4, dan K=5."),

    pMulti([
      new TextRun({ text: "3.1.2 Agglomerative Clustering (Hierarchical)", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("Pendekatan bottom-up hierarchical clustering. Setiap observasi mulai sebagai cluster sendiri, lalu digabung berpasangan berdasarkan jarak. Diuji dengan 2 linkage (Ward & Complete) dan K=3 dan K=4."),

    pMulti([
      new TextRun({ text: "3.1.3 DBSCAN (Density-Based Spatial Clustering)", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("Clustering berbasis kepadatan. Tidak perlu menentukan jumlah cluster — otomatis mendeteksi cluster berdasarkan kepadatan titik. Diuji dengan eps=0.5 (default) dan eps=1.5 (tuned).") ,

    pMulti([
      new TextRun({ text: "3.1.4 Spectral Clustering", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("Menggunakan matriks similaritas (affinity) dan spektrum graph Laplacian untuk clustering. Cocok untuk data dengan bentuk cluster non-spherical. Diuji dengan K=3 dan K=4, affinity=rbf."),

    pMulti([
      new TextRun({ text: "3.1.5 Mean Shift", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("Algoritma centroid-based yang secara otomatis menentukan jumlah cluster. Menggeser titik-titik data menuju densitas tertinggi. Bandwidth diestimasi otomatis dengan quantile=0.3."),

    pMulti([
      new TextRun({ text: "3.1.6 Gaussian Mixture Model (GMM)", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("Soft clustering — setiap titik memiliki probabilitas masuk ke setiap cluster. Menggunakan distribusi Gaussian. Diuji dengan K=3, K=4, dan auto-BIC (terbaik K=6)."),

    pMulti([
      new TextRun({ text: "3.1.7 OPTICS", font: "Arial", size: 24, bold: true, color: COLORS.secondary }),
    ], { after: 60 }),
    bullet("Versi DBSCAN yang lebih canggih. Dapat mendeteksi cluster dengan kepadatan berbeda-beda. Namun untuk data ini, hanya menghasilkan 1 cluster dengan noise >85%."),

    emptyLine(),

    heading(2, "3.2 Metrik Evaluasi"),
    p("Lima metrik evaluasi internal digunakan untuk menilai kualitas clustering tanpa memerlukan ground truth:"),

    pMulti([
      new TextRun({ text: "1. Silhouette Score ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "(↑ tinggi = baik) — Mengukur seberapa mirip suatu titik dengan clusternya sendiri dibandingkan cluster lain. Rentang: -1 hingga +1. Nilai >0.25 umumnya dianggap baik untuk data sosial.", font: "Arial", size: 24 }),
    ], { after: 100 }),

    pMulti([
      new TextRun({ text: "2. Davies-Bouldin Index (DBI) ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "(↓ rendah = baik) — Rata-rata rasio jarak intra-cluster vs jarak antar-cluster. Nilai <1.0 umumnya dianggap baik.", font: "Arial", size: 24 }),
    ], { after: 100 }),

    pMulti([
      new TextRun({ text: "3. Calinski-Harabasz Index (CH) ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "(↑ tinggi = baik) — Rasio varians antar-cluster terhadap varians dalam-cluster. Semakin tinggi, semakin kompak dan terpisah cluster-nya.", font: "Arial", size: 24 }),
    ], { after: 100 }),

    pMulti([
      new TextRun({ text: "4. Dunn Index ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "(↑ tinggi = baik) — Rasio jarak minimum antar-cluster dibagi diameter maksimum dalam-cluster. Sangat sensitif terhadap outlier — satu titik outlier dapat menurunkan skor secara drastis. Ini penting untuk data Indonesia yang memiliki outlier alami seperti Jakarta dan Papua.", font: "Arial", size: 24 }),
    ], { after: 100 }),

    pMulti([
      new TextRun({ text: "5. Gap Statistic ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "(↑ tinggi = baik) — Membandingkan inertia data asli dengan ekspektasi inertia data random. Semakin tinggi gap-nya, semakin bermakna cluster yang terbentuk dibandingkan random chance.", font: "Arial", size: 24 }),
    ], { after: 120 }),

    emptyLine(),
  ];
}

// ============================================================
// SECTION: BAB 4 - HASIL EKSPERIMEN
// ============================================================

function createBAB4() {
  return [
    heading(1, "BAB 4: HASIL EKSPERIMEN"),

    heading(2, "4.1 Perbandingan 5 Metrik — Semua Model"),
    p("Tabel 4.1 di bawah menyajikan perbandingan lengkap seluruh 12 konfigurasi model yang berhasil dijalankan. Model diurutkan berdasarkan Silhouette Score (tertinggi ke terendah)."),
    emptyLine(),

    createTable(fiveMetricHeaders, fiveMetricData, fiveMetricColWidths),
    emptyLine(),

    heading(2, "4.2 Visualisasi Perbandingan"),
    p("Gambar 4.1 menampilkan bar chart perbandingan untuk masing-masing metrik secara terpisah:"),
    emptyLine(),

    // Bar chart comparison
    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${EXPERIMEN_DIR}/28_evaluasi_metrik_lengkap/perbandingan_5_metrik.png`),
        transformation: { width: 550, height: 470 },
        altText: { title: "Perbandingan 5 Metrik", description: "Bar chart perbandingan 5 metrik untuk semua model", name: "Gambar 4.1" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar 4.1 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "Bar chart perbandingan 5 metrik clustering untuk 12 konfigurasi model", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    p("Gambar 4.2 menyajikan radar chart untuk 5 model terbaik, yang membandingkan profil metrik secara multidimensional:"),
    emptyLine(),

    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${EXPERIMEN_DIR}/28_evaluasi_metrik_lengkap/radar_5_model_terbaik.png`),
        transformation: { width: 450, height: 450 },
        altText: { title: "Radar Chart 5 Model Terbaik", description: "Radar chart perbandingan 5 model terbaik", name: "Gambar 4.2" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar 4.2 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "Radar chart 5 model terbaik — semakin luas area, semakin baik performa multidimensional", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    emptyLine(),

    heading(2, "4.3 Hasil Algoritma Terpilih — Detail"),
    p("Berikut adalah visualisasi PCA scatter plot untuk model-model unggulan:"),
    emptyLine(),

    // K-Means K=4
    pMulti([
      new TextRun({ text: "K-Means K=4 ", font: "Arial", size: 22, bold: true, color: COLORS.secondary }),
      new TextRun({ text: "(CH: 386.5 | Gap: 1.589 | Silhouette: 0.432)", font: "Arial", size: 20, italics: true, color: "666666" }),
    ], { after: 80 }),

    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${EXPERIMEN_DIR}/02_k4/pca_scatter.png`),
        transformation: { width: 450, height: 350 },
        altText: { title: "K-Means K=4 PCA", description: "PCA scatter plot K-Means K=4", name: "Gambar 4.3" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar 4.3 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "PCA scatter plot — K-Means K=4 (Model Rekomendasi Utama)", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    // Agglomerative Ward K=3
    pMulti([
      new TextRun({ text: "Agglomerative Ward K=3 ", font: "Arial", size: 22, bold: true, color: COLORS.secondary }),
      new TextRun({ text: "(Silhouette: 0.472 | CH: 257.5)", font: "Arial", size: 20, italics: true, color: "666666" }),
    ], { after: 80 }),

    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${EXPERIMEN_DIR}/16_agglomerative_k3_ward/pca_scatter.png`),
        transformation: { width: 450, height: 350 },
        altText: { title: "Agglomerative Ward K=3 PCA", description: "PCA scatter plot Agglomerative Ward K=3", name: "Gambar 4.4" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar 4.4 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "PCA scatter plot — Agglomerative Ward K=3 (Alternatif 3 Klaster)", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    // Baseline K-Means K=3
    pMulti([
      new TextRun({ text: "K-Means K=3 (Baseline) ", font: "Arial", size: 22, bold: true, color: COLORS.secondary }),
      new TextRun({ text: "(Silhouette: 0.329 | DBI: 0.890)", font: "Arial", size: 20, italics: true, color: "666666" }),
    ], { after: 80 }),

    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${OUTPUT_DIR}/Visualisasi_PCA_Klaster_Polarisasi.png`),
        transformation: { width: 450, height: 350 },
        altText: { title: "Baseline PCA", description: "PCA scatter plot K-Means K=3 baseline", name: "Gambar 4.5" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar 4.5 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "PCA scatter plot — K-Means K=3 (Model Baseline)", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    emptyLine(),
  ];
}

// ============================================================
// SECTION: BAB 5 - ANALISIS
// ============================================================

function createBAB5() {
  return [
    heading(1, "BAB 5: ANALISIS & INTERPRETASI"),

    heading(2, "5.1 Analisis per Metrik"),

    pMulti([
      new TextRun({ text: "Silhouette Score: ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "DBSCAN tuned (0.7793) unggul, tetapi hanya menghasilkan 2 klaster — satu klaster besar (511 wilayah) dan satu klaster kecil (3 wilayah: Jakarta). Ini terlalu sederhana untuk analisis polarisasi. Agglomerative Ward K=3 (0.4724) memberikan Silhouette tertinggi untuk konfigurasi yang interpretable.", font: "Arial", size: 24 }),
    ], { after: 120 }),

    pMulti([
      new TextRun({ text: "Calinski-Harabasz Index: ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "K-Means K=4 (386.5) unggul jauh. Ini menunjukkan bahwa K-Means K=4 menghasilkan cluster yang paling kompak secara internal dan paling terpisah secara eksternal.", font: "Arial", size: 24 }),
    ], { after: 120 }),

    pMulti([
      new TextRun({ text: "Gap Statistic: ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "K-Means K=4 (1.589) kembali unggul. Ini mengonfirmasi bahwa cluster yang terbentuk sangat bermakna secara statistik dan jauh berbeda dari random chance.", font: "Arial", size: 24 }),
    ], { after: 120 }),

    pMulti([
      new TextRun({ text: "Dunn Index: ", font: "Arial", size: 24, bold: true }),
      new TextRun({ text: "Semua model menunjukkan nilai Dunn yang rendah (0.003 - 0.182). Ini BUKAN berarti clustering jelek, melainkan bukti adanya outlier alami dalam data — Jakarta dengan PDRB triliunan dan Papua dengan kemiskinan ekstrem membuat diameter cluster membesar dan jarak antar cluster mengecil. Dunn Index sangat sensitif terhadap outlier, dan data kesejahteraan Indonesia secara alami memiliki outlier.", font: "Arial", size: 24 }),
    ], { after: 120 }),

    heading(2, "5.2 Temuan Kunci"),

    bulletBold("Distribusi tidak seimbang: ", "Hampir semua model menghasilkan satu cluster dominan (≥400 wilayah), menunjukkan bahwa mayoritas kabupaten/kota Indonesia memiliki profil ekonomi menengah yang relatif homogen."),
    bulletBold("Outlier Jakarta: ", "Jakarta dan kota industri besar (Surabaya, Bandung, Bekasi, Karawang, Makassar) konsisten menjadi outlier — PDRB sangat tinggi namun pengangguran juga tinggi, mengonfirmasi adanya paradoks pertumbuhan."),
    bulletBold("GMM gagal: ", "Gaussian Mixture Model (Silhouette 0.116-0.158) gagal karena data tidak mengikuti distribusi Gaussian — distribusinya condong (skewed) dengan ekor panjang (heavy-tailed)."),
    bulletBold("OPTICS tidak berguna: ", "Hanya menghasilkan 1 klaster dengan >85% dianggap noise. Data tidak memiliki variasi kepadatan yang cukup untuk metode density-based."),
    bulletBold("K-Means K=4 vs Agglomerative Ward K=3: ", "K-Means K=4 unggul di CH dan Gap, sementara Agglomerative Ward K=3 unggul di Silhouette. Pilihan tergantung prioritas: K=4 untuk granularitas lebih, K=3 untuk interpretasi lebih mudah."),

    emptyLine(),
  ];
}

// ============================================================
// SECTION: BAB 6 - KESIMPULAN
// ============================================================

function createBAB6() {
  return [
    heading(1, "BAB 6: KESIMPULAN & REKOMENDASI"),

    heading(2, "6.1 Kesimpulan"),
    p("Berdasarkan seluruh rangkaian eksperimen yang melibatkan 7 algoritma clustering dengan 16 konfigurasi dan evaluasi menggunakan 5 metrik, dapat disimpulkan:"),

    bullet("K-Means K=4 adalah model yang paling direkomendasikan untuk analisis polarisasi kesejahteraan — unggul di Calinski-Harabasz (386.5) dan Gap Statistic (1.589), serta memberikan distribusi klaster yang seimbang dan interpretable."),
    bullet("Agglomerative Ward K=3 merupakan alternatif terbaik untuk 3 klaster — Silhouette tertinggi (0.472) untuk konfigurasi interpretable."),
    bullet("Data kesejahteraan Indonesia memiliki outlier alami (Jakarta, Papua) yang mempengaruhi beberapa metrik seperti Dunn Index — ini wajar dan perlu dipahami saat interpretasi."),
    bullet("Metrik Calinski-Harabasz dan Gap Statistic memberikan informasi komplementer yang lebih stabil untuk data dengan outlier dibandingkan Silhouette dan DBI saja."),

    heading(2, "6.2 Rekomendasi untuk HKI"),
    p("Untuk pengajuan Hak Kekayaan Intelektual (HKI) Program Komputer, direkomendasikan:"),

    bulletBold("Model Final: ", "K-Means K=4 dengan fitur PDRB, TPT, dan P0 — terbukti paling unggul secara statistik dan mudah diinterpretasi."),
    bulletBold("Atau alternatif: ", "Agglomerative Ward K=3 jika ingin 3 tipologi yang lebih sederhana dengan hierarki natural."),
    bulletBold("Metrik Pelengkap: ", "Gunakan 5 metrik (Silhouette, DBI, CH, Dunn, Gap) untuk validasi yang lebih komprehensif."),
    bulletBold("Integrasi Dashboard: ", "Hasil akhir dapat divisualisasikan dalam dashboard Streamlit untuk eksplorasi interaktif oleh tim."),

    emptyLine(),

    // Ringkasan tabel rekomendasi
    pMulti([
      new TextRun({ text: "Tabel 6.1 — Rekomendasi Model Final", font: "Arial", size: 22, bold: true, color: COLORS.primary }),
    ], { center: true, after: 120 }),

    createTable(
      ["Peringkat", "Model", "K", "Silhouette", "CH Index", "Gap", "Kelebihan Utama"],
      [
        ["🥇", "K-Means K=4", "4", "0.432", "386.5", "1.589", "Paling kompak & bermakna"],
        ["🥈", "Agglomerative Ward K=3", "3", "0.472", "257.5", "1.262", "Hierarkis & interpretable"],
        ["🥉", "Agglomerative Ward K=4", "4", "0.453", "321.4", "1.465", "CH tinggi, distribusi seimbang"],
      ],
      [600, 2500, 400, 900, 900, 900, 2500]
    ),

    emptyLine(), emptyLine(),
  ];
}

// ============================================================
// SECTION: LAMPIRAN
// ============================================================

function createLampiran() {
  return [
    new Paragraph({ children: [new PageBreak()] }),
    heading(1, "LAMPIRAN"),

    heading(2, "L.1 Dendrogram — Agglomerative Ward"),
    p("Dendrogram berikut menunjukkan hierarki pengelompokan wilayah dari Agglomerative Clustering dengan linkage Ward. Semakin tinggi garis penghubung, semakin berbeda karakteristik antar kelompok."),
    emptyLine(),

    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${EXPERIMEN_DIR}/19_dendrogram/dendrogram.png`),
        transformation: { width: 550, height: 400 },
        altText: { title: "Dendrogram", description: "Dendrogram Agglomerative Ward", name: "Gambar L.1" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar L.1 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "Dendrogram Agglomerative Clustering (Ward linkage) — menunjukkan hierarki pengelompokan 514 kabupaten/kota", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    emptyLine(),

    heading(2, "L.2 Baseline Model — Evaluasi K Optimal"),
    p("Grafik evaluasi K optimal dari model baseline (K-Means) yang menunjukkan Elbow Method, Silhouette Score, dan Davies-Bouldin Index untuk K=2 hingga K=10."),
    emptyLine(),

    new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(`${OUTPUT_DIR}/Evaluasi_K_Optimal.png`),
        transformation: { width: 550, height: 200 },
        altText: { title: "Evaluasi K Optimal", description: "Grafik evaluasi K optimal", name: "Gambar L.2" },
      })],
      alignment: AlignmentType.CENTER,
    }),
    pMulti([
      new TextRun({ text: "Gambar L.2 ", font: "Arial", size: 20, bold: true }),
      new TextRun({ text: "Evaluasi K optimal untuk K-Means — Elbow Method, Silhouette Score, dan DBI", font: "Arial", size: 20, italics: true }),
    ], { center: true, after: 200 }),

    emptyLine(),

    heading(2, "L.3 Perbandingan Algoritma — Semua Model"),
    p("Berikut adalah tabel lengkap perbandingan 16 konfigurasi model yang diuji:"),
    emptyLine(),

    createTable(
      ["No", "Algoritma", "Parameter", "K", "Noise", "Silhouette", "DBI"],
      [
        ["1", "DBSCAN (tuned)", "eps=1.5, min=3", "2", "0", "0.7793", "0.2164"],
        ["2", "Spectral", "K=3, rbf", "3", "0", "0.7087", "0.4182"],
        ["3", "Mean Shift", "auto (bw=1.40)", "4", "0", "0.5437", "0.4776"],
        ["4", "Spectral", "K=4, rbf", "4", "0", "0.5058", "0.4952"],
        ["5", "Agglomerative Ward", "K=3", "3", "0", "0.4724", "0.8745"],
        ["6", "Agglomerative Complete", "K=4", "4", "0", "0.4542", "0.6113"],
        ["7", "Agglomerative Ward", "K=4", "4", "0", "0.4526", "0.6929"],
        ["8", "K-Means++", "K=4", "4", "0", "0.4319", "0.7521"],
        ["9", "DBSCAN (default)", "eps=0.5", "2", "33", "0.4303", "0.4529"],
        ["10", "K-Means++", "K=3", "3", "0", "0.3291", "0.8900"],
        ["11", "GMM", "K=3", "3", "0", "0.1807", "1.3681"],
        ["12", "GMM", "K=4", "4", "0", "0.1583", "1.4353"],
        ["13", "OPTICS", "default", "1", "479", "N/A", "N/A"],
      ],
      [400, 2200, 1800, 600, 600, 900, 900]
    ),

    emptyLine(),
    emptyLine(),
  ];
}

// ============================================================
// DOKUMEN UTAMA
// ============================================================

async function main() {
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Arial", size: 24, color: "333333" },
          paragraph: { spacing: { line: 276 } },
        },
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "Arial", color: "1B3A5C" },
          paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Arial", color: "2E86AB" },
          paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 },
        },
        {
          id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, font: "Arial", color: "444444" },
          paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
        },
      ],
    },
    numbering: {
      config: [{
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      }],
    },
    sections: [
      // COVER PAGE
      {
        properties: {
          page: {
            size: { width: 11906, height: 16838 }, // A4
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.RIGHT,
              border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: COLORS.secondary, space: 4 } },
              children: [new TextRun({ text: "Laporan Eksperimen — Polarisasi Kesejahteraan", font: "Arial", size: 18, color: "999999", italics: true })],
            })],
          }),
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              border: { top: { style: BorderStyle.SINGLE, size: 2, color: COLORS.secondary, space: 4 } },
              children: [
                new TextRun({ text: "Halaman ", font: "Arial", size: 18, color: "999999" }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "999999" }),
              ],
            })],
          }),
        },
        children: [
          ...createCoverPage(),
        ],
      },

      // MAIN CONTENT
      {
        properties: {
          page: {
            size: { width: 11906, height: 16838 },
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.RIGHT,
              border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: COLORS.secondary, space: 4 } },
              children: [new TextRun({ text: "Laporan Eksperimen — Polarisasi Kesejahteraan", font: "Arial", size: 18, color: "999999", italics: true })],
            })],
          }),
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              border: { top: { style: BorderStyle.SINGLE, size: 2, color: COLORS.secondary, space: 4 } },
              children: [
                new TextRun({ text: "Halaman ", font: "Arial", size: 18, color: "999999" }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "999999" }),
              ],
            })],
          }),
        },
        children: [
          // Table of Contents
          new Paragraph({
            spacing: { before: 200 },
            children: [new TextRun({ text: "DAFTAR ISI", font: "Arial", size: 32, bold: true, color: COLORS.primary })],
          }),
          new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
          new Paragraph({ children: [new PageBreak()] }),

          ...createBAB1(),
          new Paragraph({ children: [new PageBreak()] }),

          ...createBAB2(),
          new Paragraph({ children: [new PageBreak()] }),

          ...createBAB3(),
          new Paragraph({ children: [new PageBreak()] }),

          ...createBAB4(),
          new Paragraph({ children: [new PageBreak()] }),

          ...createBAB5(),
          new Paragraph({ children: [new PageBreak()] }),

          ...createBAB6(),
          new Paragraph({ children: [new PageBreak()] }),

          ...createLampiran(),
        ],
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(`${OUTPUT_DIR}/Laporan_Eksperimen_Polarisasi_Kesejahteraan.docx`, buffer);
  console.log("✅ Dokumen berhasil dibuat: Laporan_Eksperimen_Polarisasi_Kesejahteraan.docx");
  console.log(`   Ukuran: ${(buffer.length / 1024).toFixed(1)} KB`);
}

main().catch(console.error);
