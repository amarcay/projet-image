"""
Chargement et prétraitement des images satellites.

Justification du prétraitement choisi :
- Les deux images présentent des luminosités différentes (t1 plus sombre).
- Étape 1 : CLAHE sur le canal L (espace LAB) pour corriger le contraste local
  sans altérer les couleurs. Cela améliore la séparabilité des classes.
- Étape 2 : correspondance d'histogramme (histogram matching) de t1 vers t0.
  Cela normalise la distribution globale des couleurs entre les deux images,
  rendant les ratios verts directement comparables pour le clustering joint.
- Aucun débruitage : les images satellites sont déjà nettes.
"""

import cv2
import numpy as np
from skimage.exposure import match_histograms


def load_image(path: str) -> np.ndarray:
    """Charge une image en BGR (OpenCV) et la convertit en RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Normalise le contraste via CLAHE sur le canal L (espace LAB).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def match_histograms_to_reference(img_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """
    Ajuste la distribution des couleurs de img_rgb pour qu'elle corresponde
    à celle de reference_rgb (canal par canal).
    Rend les deux images photométriquement comparables.
    """
    matched = match_histograms(img_rgb, reference_rgb, channel_axis=-1)
    return matched.astype(np.uint8)


def to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """Convertit une image RGB en HSV (float32, H∈[0,360], S∈[0,1], V∈[0,1])."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)


def load_and_preprocess(path: str) -> dict:
    """
    Chargement + prétraitement d'une image seule (CLAHE uniquement).
    Pour la normalisation inter-images, utiliser load_and_preprocess_pair().
    """
    raw = load_image(path)
    preprocessed = normalize_clahe(raw)
    hsv = to_hsv(preprocessed)
    return {"raw": raw, "preprocessed": preprocessed, "hsv": hsv}


def load_and_preprocess_pair(path_t0: str, path_t1: str) -> tuple[dict, dict]:
    """
    Charge et prétraite la paire d'images de manière cohérente :
      1. CLAHE sur chaque image
      2. Histogram matching : t1 est normalisée sur t0

    Retourne (data_t0, data_t1), chaque dict contenant : raw, preprocessed, hsv.
    """
    raw_t0 = load_image(path_t0)
    raw_t1 = load_image(path_t1)

    clahe_t0 = normalize_clahe(raw_t0)
    clahe_t1 = normalize_clahe(raw_t1)

    # Normalise t1 sur t0 pour rendre les couleurs comparables
    matched_t1 = match_histograms_to_reference(clahe_t1, clahe_t0)

    data_t0 = {"raw": raw_t0, "preprocessed": clahe_t0, "hsv": to_hsv(clahe_t0)}
    data_t1 = {"raw": raw_t1, "preprocessed": matched_t1, "hsv": to_hsv(matched_t1)}
    return data_t0, data_t1
