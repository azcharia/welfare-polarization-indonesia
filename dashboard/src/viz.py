"""
src/viz.py
Modul untuk visualisasi dashboard — PCA scatter, cluster profiles, bar charts, map.
Semua visualisasi menggunakan Plotly untuk interaktivitas.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import FEATURE_COLS, FEATURE_SHORT


# ============================================================
# WARNA KLASTER
# ============================================================
CLUSTER_COLORS = {
    "Kerentanan Struktural": "#E8505B",
    "Episentrum Ekonomi Jenuh": "#2E86AB",
    "Wilayah Transisi": "#F18F01",
    "Kelas Menengah Atas": "#2CA02C",
}

# Default color sequence
COLOR_SEQ = px.colors.qualitative.Set2


def plot_pca_scatter(df_pca, explained_var, tipologi_colors=None):
    """
    PCA Scatter plot interaktif dengan Plotly.
    Hover menampilkan nama kab/kota, provinsi, dan nilai fitur.
    """
    if tipologi_colors is None:
        tipologi_colors = CLUSTER_COLORS

    # Color mapping
    unique_tipologi = df_pca['Tipologi'].unique() if 'Tipologi' in df_pca.columns else df_pca['Klaster'].unique()
    color_discrete_map = {}
    for i, t in enumerate(sorted(unique_tipologi)):
        if t in tipologi_colors:
            color_discrete_map[t] = tipologi_colors[t]
        else:
            color_discrete_map[t] = COLOR_SEQ[i % len(COLOR_SEQ)]

    # Hover data
    hover_data = {
        'Kab/Kota': True,
        'Provinsi': True,
        'PDRB': True,
        'TPT': True,
        'P0': True,
        'PC1': False,
        'PC2': False,
    }

    fig = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color='Tipologi' if 'Tipologi' in df_pca.columns else 'Klaster',
        color_discrete_map=color_discrete_map if 'Tipologi' in df_pca.columns else None,
        hover_name='Kab/Kota',
        hover_data={
            'Provinsi': True,
            'PDRB': ':.2f',
            'TPT': ':.2f',
            'P0': ':.2f',
            'PC1': False,
            'PC2': False,
        },
        labels={
            'PC1': f'Principal Component 1 ({explained_var[0]*100:.1f}%)',
            'PC2': f'Principal Component 2 ({explained_var[1]*100:.1f}%)',
        },
        title="Visualisasi Klaster — PCA 2D",
        opacity=0.8,
        size_max=12,
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.5, color='white')),
        selector=dict(mode='markers'),
    )

    fig.update_layout(
        title=dict(
            text="Visualisasi Klaster — PCA 2D",
            font=dict(size=14, color="#FFFFFF", family="Outfit")
        ),
        legend_title_text='Tipologi Wilayah',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#E0E0E0")
        ),
        margin=dict(l=20, r=20, t=60, b=80),
        hoverlabel=dict(
            bgcolor="rgba(28, 33, 39, 0.95)",
            font_size=12,
            font_color="#ffffff",
            font_family="Plus Jakarta Sans"
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(128,128,128,0.15)',
            tickfont=dict(color='#B0B0B0'),
            title=dict(font=dict(color='#E0E0E0'))
        ),
        yaxis=dict(
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(128,128,128,0.15)',
            tickfont=dict(color='#B0B0B0'),
            title=dict(font=dict(color='#E0E0E0'))
        ),
        height=500,
    )

    return fig


def plot_cluster_radar(profil):
    """
    Radar chart perbandingan karakteristik antar klaster.
    Menampilkan PDRB, TPT, dan P0 (dinormalisasi).
    """
    # Normalisasi untuk radar chart
    radar_data = profil.copy()
    pdrb_mean_col = [c for c in radar_data.columns if 'PDRB_mean' in c][0]
    tpt_mean_col = [c for c in radar_data.columns if 'TPT_mean' in c][0]
    p0_mean_col = [c for c in radar_data.columns if 'P0_mean' in c][0]

    # Normalize 0-1 for radar
    for col in [pdrb_mean_col, tpt_mean_col, p0_mean_col]:
        max_val = radar_data[col].max()
        if max_val > 0:
            radar_data[f'{col}_norm'] = radar_data[col] / max_val
        else:
            radar_data[f'{col}_norm'] = 0

    categories = ['PDRB', 'TPT', 'P0 (Kemiskinan)']

    fig = go.Figure()

    for _, row in radar_data.iterrows():
        klaster = int(row.get('Klaster', 0))
        tipologi = row.get('Tipologi', f'Klaster {klaster}')
        values = [
            row.get(f'{pdrb_mean_col}_norm', 0),
            row.get(f'{tpt_mean_col}_norm', 0),
            row.get(f'{p0_mean_col}_norm', 0),
        ]
        values += values[:1]  # Close the radar

        color = CLUSTER_COLORS.get(tipologi, COLOR_SEQ[klaster % len(COLOR_SEQ)])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=tipologi,
            line=dict(color=color, width=2),
            fillcolor=color,
            fill='toself',
            opacity=0.3,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0.2', '0.4', '0.6', '0.8', '1.0'],
                gridcolor='rgba(128,128,128,0.2)',
                linecolor='rgba(128,128,128,0.2)',
                tickfont=dict(color='#B0B0B0'),
            ),
            angularaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                linecolor='rgba(128,128,128,0.2)',
                tickfont=dict(color='#E0E0E0'),
            )
        ),
        title_text="Profil Klaster — Radar Chart (Normalized)",
        title_font=dict(size=14, color="#FFFFFF", family="Outfit"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#E0E0E0")
        ),
        margin=dict(l=80, r=80, t=80, b=100),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def plot_cluster_bars(profil):
    """
    Grouped bar chart perbandingan nilai rata-rata per klaster.
    """
    pdrb_mean_col = [c for c in profil.columns if 'PDRB_mean' in c][0]
    tpt_mean_col = [c for c in profil.columns if 'TPT_mean' in c][0]
    p0_mean_col = [c for c in profil.columns if 'P0_mean' in c][0]

    # PDRB dalam Juta untuk tampilan
    df_plot = profil.copy()
    df_plot['PDRB (Juta)'] = df_plot[pdrb_mean_col] / 1e6
    df_plot['Tipologi'] = df_plot.get('Tipologi', df_plot['Klaster'].astype(str))
 
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('PDRB (Juta Rp)', 'TPT (%)', 'P0 - Kemiskinan (%)'),
        horizontal_spacing=0.1,
    )
 
    for _, row in df_plot.iterrows():
        tipologi = row.get('Tipologi', f"Klaster {int(row['Klaster'])}")
        color = CLUSTER_COLORS.get(tipologi, COLOR_SEQ[int(row['Klaster']) % len(COLOR_SEQ)])
        jumlah = int(row.get('Jumlah', 0))
 
        fig.add_trace(
            go.Bar(name=tipologi, x=[tipologi], y=[row['PDRB (Juta)']],
                   marker_color=color, showlegend=True,
                   text=f"Rp{row['PDRB (Juta)']:.1f} Jt" if row['PDRB (Juta)'] >= 10 else f"Rp{row['PDRB (Juta)']:.2f} Jt", textposition='outside'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name=tipologi, x=[tipologi], y=[row[tpt_mean_col]],
                   marker_color=color, showlegend=False,
                   text=f"{row[tpt_mean_col]:.2f}%", textposition='outside'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name=tipologi, x=[tipologi], y=[row[p0_mean_col]],
                   marker_color=color, showlegend=False,
                   text=f"{row[p0_mean_col]:.2f}%", textposition='outside'),
            row=1, col=3
        )

    fig.update_layout(
        title_text="Perbandingan Karakteristik per Klaster",
        title_font=dict(size=14, color="#FFFFFF", family="Outfit"),
        height=400,
        margin=dict(l=20, r=20, t=80, b=120),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#E0E0E0")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#B0B0B0')),
        yaxis=dict(tickfont=dict(color='#B0B0B0'), showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
        xaxis2=dict(tickfont=dict(color='#B0B0B0')),
        yaxis2=dict(tickfont=dict(color='#B0B0B0'), showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
        xaxis3=dict(tickfont=dict(color='#B0B0B0')),
        yaxis3=dict(tickfont=dict(color='#B0B0B0'), showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
    )

    # Pastikan judul subplot berkontras tinggi (putih) di dark mode
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color='#E0E0E0', family="Outfit")

    return fig


def plot_tipologi_distribution(df):
    """
    Pie / donut chart distribusi jumlah wilayah per tipologi.
    """
    if 'Tipologi' not in df.columns:
        return None

    counts = df['Tipologi'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,
        marker=dict(colors=[CLUSTER_COLORS.get(t, '#999999') for t in counts.index]),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Jumlah: %{value} wilayah<br>Persentase: %{percent}<extra></extra>',
    )])

    fig.update_layout(
        title="Distribusi Wilayah per Tipologi",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    return fig


def plot_metrics_comparison(metrics_df):
    """
    Bar chart perbandingan metrik untuk semua model.
    metrics_df: DataFrame dengan kolom Model, Silhouette, DBI, CH
    """
    if metrics_df is None or metrics_df.empty:
        return None

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Silhouette Score ↑', 'Davies-Bouldin Index ↓', 'Calinski-Harabasz ↑'),
        horizontal_spacing=0.08,
    )

    models = metrics_df['Model'].tolist()
    colors = [COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(len(models))]

    # Silhouette
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['Silhouette'], marker_color=colors,
               text=[f"{v:.3f}" for v in metrics_df['Silhouette']], textposition='outside'),
        row=1, col=1
    )

    # DBI (lower is better)
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['DBI'], marker_color=colors,
               text=[f"{v:.3f}" for v in metrics_df['DBI']], textposition='outside'),
        row=1, col=2
    )

    # CH
    fig.add_trace(
        go.Bar(x=models, y=metrics_df['CH'], marker_color=colors,
               text=[f"{v:.1f}" for v in metrics_df['CH']], textposition='outside'),
        row=1, col=3
    )

    fig.update_layout(
        title_text="Perbandingan Metrik Antar Model",
        title_font=dict(size=14, color="#FFFFFF", family="Outfit"),
        height=400,
        margin=dict(l=20, r=20, t=80, b=120),
        showlegend=False,
        xaxis_tickangle=-45,
        xaxis2_tickangle=-45,
        xaxis3_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#B0B0B0')),
        yaxis=dict(tickfont=dict(color='#B0B0B0'), showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
        xaxis2=dict(tickfont=dict(color='#B0B0B0')),
        yaxis2=dict(tickfont=dict(color='#B0B0B0'), showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
        xaxis3=dict(tickfont=dict(color='#B0B0B0')),
        yaxis3=dict(tickfont=dict(color='#B0B0B0'), showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
    )

    # Pastikan judul subplot berkontras tinggi (putih) di dark mode
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color='#E0E0E0', family="Outfit")

    # Update y-axis ranges
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score (lower better)", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=3)

    return fig


def format_dataframe_display(df):
    """
    Format dataframe untuk ditampilkan di Streamlit dengan styling.
    Menambahkan indikator emoji visual premium untuk membedakan tipologi secara instan.
    """
    display_cols = ['Provinsi', 'Kab/Kota']
    if 'Tipologi' in df.columns:
        display_cols.append('Tipologi')
    if 'Klaster' in df.columns:
        display_cols.append('Klaster')

    display_cols.extend(FEATURE_COLS)

    df_display = df[display_cols].copy()
    df_display = df_display.rename(columns=FEATURE_SHORT)

    # Map nama Tipologi ke emoji visual yang sangat jelas di light/dark mode
    emoji_map = {
        "Episentrum Ekonomi Jenuh": "🔵 Episentrum Ekonomi Jenuh",
        "Kelas Menengah Atas": "🟢 Kelas Menengah Atas",
        "Wilayah Transisi": "🟡 Wilayah Transisi",
        "Kerentanan Struktural": "🔴 Kerentanan Struktural",
    }
    
    if 'Tipologi' in df_display.columns:
        df_display['Tipologi'] = df_display['Tipologi'].map(lambda val: emoji_map.get(val, val))

    return df_display
