"""
Exploratory Data Analysis visualizations for YouTube AdView prediction.

Usage
-----
    from src.eda import plot_category_distribution, plot_adview_distribution, plot_correlation_heatmap

    df = pd.read_csv("data/train.csv")
    plot_category_distribution(df)
    plot_adview_distribution(df)
    plot_correlation_heatmap(df)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_category_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Histogram of video category distribution."""
    plt.figure(figsize=(8, 5))
    plt.hist(df["category"], bins=8, color="steelblue", edgecolor="white")
    plt.title("Video Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_adview_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Line plot of adview values to inspect distribution and outliers."""
    plt.figure(figsize=(10, 4))
    plt.plot(df["adview"].values, color="steelblue", linewidth=0.5)
    plt.title("AdView Distribution (raw)")
    plt.xlabel("Sample Index")
    plt.ylabel("AdViews")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Correlation heatmap of all numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=np.zeros_like(corr, dtype=bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=True,
        fmt=".2f",
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: str | None) -> None:
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()
