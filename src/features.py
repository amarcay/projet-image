"""
Construction du vecteur de features par pixel.

Features retenues :
  - RGB normalisé [0,1]         → couleur brute
  - HSV                         → teinte (discriminant végétation vs sol)
  - Ratio vert G/(R+G+B)        → indice de végétation simple
  - Moyenne locale (fenêtre 5)  → contexte spatial
  - Variance locale (fenêtre 5) → texture (forêt = haute variance)
  - Magnitude Sobel             → gradient local (bords/texture végétation)
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter


def green_ratio(img_rgb: np.ndarray) -> np.ndarray:
    """Calcule G/(R+G+B) pour chaque pixel. Retourne un tableau 2D float32."""
    img = img_rgb.astype(np.float32)
    total = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    # Évite la division par zéro sur les pixels noirs
    ratio = np.where(total > 0, img[:, :, 1] / total, 0.0)
    return ratio.astype(np.float32)


def local_mean_variance(img_rgb: np.ndarray, size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la moyenne et la variance locales sur le canal vert (fenêtre carrée).
    Utilise E[X²] - E[X]² via uniform_filter (scipy.ndimage).
    size=5 → fenêtre 5×5 pixels.
    """
    green = img_rgb[:, :, 1].astype(np.float32) / 255.0
    loc_mean = uniform_filter(green, size=size).astype(np.float32)
    loc_var = (uniform_filter(green ** 2, size=size) - loc_mean ** 2).astype(np.float32)
    loc_var = np.clip(loc_var, 0, None)  # évite les valeurs négatives dues aux arrondis flottants
    v_max = loc_var.max()
    if v_max > 0:
        loc_var /= v_max
    return loc_mean, loc_var


def sobel_magnitude(img_rgb: np.ndarray) -> np.ndarray:
    """
    Calcule la magnitude du gradient de Sobel sur le canal vert.

    Justification : les arbres et arbustes présentent des contours internes
    nombreux (feuilles, branches) qui génèrent une forte magnitude de gradient,
    contrairement au sol nu ou aux routes qui sont texturalement homogènes.
    Le canal vert est choisi car c'est le plus discriminant pour la végétation.
    """
    green = img_rgb[:, :, 1].astype(np.float32)
    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # Normalise entre 0 et 1
    m_max = magnitude.max()
    if m_max > 0:
        magnitude /= m_max
    return magnitude.astype(np.float32)


def build_feature_matrix(img_rgb: np.ndarray, img_hsv: np.ndarray) -> np.ndarray:
    """
    Assemble toutes les features en une matrice (N_pixels, N_features).

    Colonnes :
      0-2  : R, G, B normalisés [0,1]
      3-5  : H normalisé [0,1], S, V
      6    : ratio vert G/(R+G+B)
      7    : moyenne locale (canal vert)
      8    : variance locale (canal vert)
      9    : magnitude Sobel (canal vert)
    """
    h, w = img_rgb.shape[:2]

    rgb_norm = img_rgb.astype(np.float32) / 255.0

    hsv_norm = img_hsv.copy()
    hsv_norm[:, :, 0] /= 360.0  # H → [0,1]

    ratio = green_ratio(img_rgb)
    loc_mean, loc_var = local_mean_variance(img_rgb)
    sobel = sobel_magnitude(img_rgb)

    feature_maps = [
        rgb_norm[:, :, 0],   # R
        rgb_norm[:, :, 1],   # G
        rgb_norm[:, :, 2],   # B
        hsv_norm[:, :, 0],   # H
        hsv_norm[:, :, 1],   # S
        hsv_norm[:, :, 2],   # V
        ratio,               # ratio vert
        loc_mean,            # moyenne locale
        loc_var,             # variance locale
        sobel,               # magnitude Sobel
    ]

    matrix = np.stack(feature_maps, axis=-1).reshape(h * w, len(feature_maps))
    return matrix.astype(np.float32)
