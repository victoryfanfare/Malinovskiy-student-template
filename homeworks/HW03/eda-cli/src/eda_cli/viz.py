from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Для числовых колонок строит по отдельной гистограмме.
    Возвращает список путей к PNG.
    """
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    for i, name in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[name].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Гистограмма с KDE
        sns.histplot(s.values, bins=bins, kde=True, ax=ax, color='skyblue', edgecolor='black')
        
        # Добавляем вертикальную линию для среднего
        ax.axvline(s.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {s.mean():.2f}')
        
        # Добавляем вертикальные линии для квантилей
        ax.axvline(s.quantile(0.25), color='green', linestyle=':', linewidth=1.5, label='Q1')
        ax.axvline(s.quantile(0.75), color='green', linestyle=':', linewidth=1.5, label='Q3')
        ax.axvline(s.median(), color='orange', linestyle='-', linewidth=1.5, label=f'Median: {s.median():.2f}')
        
        ax.set_title(f"Histogram of {name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем текстовую статистику
        stats_text = f"n={len(s):,}\nμ={s.mean():.2f}\nσ={s.std():.2f}\nmin={s.min():.2f}\nmax={s.max():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.tight_layout()

        out_path = out_dir / f"hist_{i+1}_{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Простая визуализация пропусков: где True=пропуск, False=значение.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Рисуем пустой график
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Empty dataset", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        mask = df.isna().values
        fig, ax = plt.subplots(figsize=(min(16, df.shape[1] * 0.6), 6))
        
        # Используем seaborn heatmap для лучшего визуального представления
        sns.heatmap(mask, cmap=['lightgreen', 'salmon'], 
                   cbar_kws={'label': 'Missing (True/False)'},
                   yticklabels=False, ax=ax)
        
        ax.set_xlabel("Columns", fontsize=12)
        ax.set_ylabel("Rows (sampled)", fontsize=12)
        ax.set_title("Missing Values Matrix", fontsize=14, fontweight='bold')
        
        # Поворачиваем подписи столбцов для лучшей читаемости
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=9)
        
        # Добавляем аннотацию с общей статистикой пропусков
        total_missing = mask.sum()
        total_cells = mask.size
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        stats_text = f"Total missing: {total_missing:,} ({missing_pct:.1f}%)\nRows: {df.shape[0]:,}, Cols: {df.shape[1]:,}"
        ax.text(0.02, -0.1, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляции числовых признаков.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation\n(need at least 2)", 
                ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(8, corr.shape[1]), max(6, corr.shape[0])))
        
        # Используем seaborn heatmap
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title("Correlation Heatmap (Pearson)", fontsize=14, fontweight='bold')
        
        # Поворачиваем подписи для лучшей читаемости
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        
        # Добавляем информацию о сильных корреляциях
        strong_correlations = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:  # сильная корреляция
                    strong_correlations.append(
                        f"{corr.columns[i]} ↔ {corr.columns[j]}: {val:.2f}"
                    )
        
        if strong_correlations:
            corr_text = "Strong correlations (|r| > 0.7):\n" + "\n".join(strong_correlations[:5])
            ax.text(0.02, -0.15, corr_text, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)
    return paths


def plot_top_categories_chart(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
    max_charts: int = 5
) -> List[Path]:
    """
    Создаёт bar charts для top категорий.
    Возвращает список путей к PNG файлам.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    
    for i, (col_name, df_top) in enumerate(list(top_cats.items())[:max_charts]):
        if df_top.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Создаём bar chart
        bars = ax.bar(range(len(df_top)), df_top['count'], color='steelblue', edgecolor='black')
        
        # Добавляем подписи значений
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:,}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Top Categories for "{col_name}"', fontsize=14, fontweight='bold')
        
        # Устанавливаем подписи на оси X
        ax.set_xticks(range(len(df_top)))
        ax.set_xticklabels(df_top['value'], rotation=45, ha='right', fontsize=10)
        
        # Добавляем вторую ось Y для процентов
        ax2 = ax.twinx()
        ax2.plot(range(len(df_top)), df_top['share'] * 100, 
                color='red', marker='o', linestyle='--', linewidth=2, markersize=6)
        ax2.set_ylabel('Percentage (%)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Добавляем сетку
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем легенду
        ax.legend(['Count'], loc='upper left')
        ax2.legend(['Percentage'], loc='upper right')
        
        fig.tight_layout()
        
        out_path = out_dir / f"top_categories_{col_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        paths.append(out_path)
    
    return paths


def plot_boxplots_numeric(
    df: pd.DataFrame,
    out_path: PathLike,
    max_columns: int = 8
) -> Path:
    """
    Создаёт боксплоты для числовых колонок.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    
    if numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No numeric columns for boxplots", 
                ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        # Ограничиваем количество колонок для визуализации
        cols_to_plot = numeric_df.columns[:max_columns]
        
        # Создаём subplot с оптимальным размером
        n_cols = len(cols_to_plot)
        fig_width = min(20, n_cols * 4)
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 6))
        
        # Если только одна колонка, axes будет не списком
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            data = numeric_df[col].dropna()
            
            # Создаём боксплот
            bp = ax.boxplot(data, patch_artist=True)
            
            # Настраиваем внешний вид
            bp['boxes'][0].set_facecolor('lightblue')
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            bp['whiskers'][0].set_color('black')
            bp['whiskers'][1].set_color('black')
            bp['caps'][0].set_color('black')
            bp['caps'][1].set_color('black')
            bp['fliers'][0].set(marker='o', color='red', alpha=0.5)
            
            # Добавляем точки данных для лучшей визуализации
            jitter = np.random.normal(1, 0.04, size=len(data))
            ax.scatter(jitter, data, alpha=0.4, color='green', s=20)
            
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Добавляем статистику
            stats_text = f"n={len(data):,}\nμ={data.mean():.2f}\nσ={data.std():.2f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Boxplots of Numeric Columns', fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path