"""
Fonctions de visualisation : histogrammes, images segmentées, cartes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram


def plot_histograms(img_rgb: np.ndarray, title: str = "", ax_row=None) -> matplotlib.figure.Figure:
    """Affiche les histogrammes des 3 canaux RGB d'une image."""
    fig = None
    if ax_row is None:
        fig, ax_row = plt.subplots(1, 3, figsize=(12, 3))

    colors = ["red", "green", "blue"]
    labels = ["R", "G", "B"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax_row[i].hist(img_rgb[:, :, i].ravel(), bins=64, color=color, alpha=0.7)
        ax_row[i].set_title(f"Canal {label}")
        ax_row[i].set_xlabel("Intensité")
        ax_row[i].set_ylabel("Fréquence")

    if title:
        ax_row[0].figure.suptitle(title, fontsize=13, fontweight="bold")

    return fig or ax_row[0].figure


def plot_dendrogram(Z: np.ndarray, k: int, title: str = "Dendrogramme CAH") -> matplotlib.figure.Figure:
    """Affiche le dendrogramme avec un trait horizontal indiquant le seuil choisi."""
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=45)

    # Seuil de coupure pour k clusters
    cutoff = (Z[-(k - 1), 2] + Z[-k, 2]) / 2
    ax.axhline(y=cutoff, color="red", linestyle="--", label=f"Coupure → k={k}")
    ax.set_title(title)
    ax.set_xlabel("Pixels (groupes)")
    ax.set_ylabel("Distance Ward")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_segmentation(img_rgb: np.ndarray, labels_2d: np.ndarray, k: int, title: str = "") -> matplotlib.figure.Figure:
    """Affiche l'image originale et la carte de segmentation côte à côte."""
    cmap = plt.cm.get_cmap("tab10", k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    im = axes[1].imshow(labels_2d, cmap=cmap, vmin=0, vmax=k - 1)
    axes[1].set_title(f"Segmentation K-means (k={k})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], ticks=range(k), label="Cluster")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_vegetation_mask(img_rgb: np.ndarray, mask: np.ndarray, title: str = "") -> matplotlib.figure.Figure:
    """Affiche l'image originale et le masque de végétation côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    overlay = img_rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 200, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title("Masque végétation (vert)")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_deforestation_map(carte: np.ndarray, stats: dict) -> matplotlib.figure.Figure:
    """Affiche la carte de déforestation avec légende et statistiques."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(carte)
    ax.axis("off")
    ax.set_title("Carte de déforestation", fontsize=14, fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor="#22228B", label="Végétation gagnée (reboisement)"),
        mpatches.Patch(facecolor="#DC3232", label="Végétation perdue (déforestation)"),
        mpatches.Patch(facecolor="#228B22", label="Végétation conservée"),
        mpatches.Patch(facecolor="#B4B4B4", label="Sol / urbain"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    info = (
        f"Surface t0 : {stats['surface_t0']:.1%}\n"
        f"Surface t1 : {stats['surface_t1']:.1%}\n"
        f"Perte relative : {stats['perte_pct']:.1f}%"
    )
    ax.text(
        0.01, 0.99, info,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    return fig
