"""
Post-traitement du masque de végétation.

Opérations morphologiques appliquées :
  1. Ouverture  : supprime les petits faux positifs (pixels isolés détectés comme végétation)
  2. Fermeture  : comble les petits trous dans les zones végétalisées détectées
"""

import cv2
import numpy as np


def build_vegetation_mask(labels_2d: np.ndarray, vegetation_clusters: int | list[int]) -> np.ndarray:
    """Construit un masque binaire uint8 (255 = végétation, 0 = non-végétation).
    Accepte un ou plusieurs clusters végétation.
    """
    if isinstance(vegetation_clusters, int):
        vegetation_clusters = [vegetation_clusters]
    mask = np.zeros(labels_2d.shape, dtype=bool)
    for c in vegetation_clusters:
        mask |= (labels_2d == c)
    return mask.astype(np.uint8) * 255


def apply_morphological_cleanup(
    mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 5,
) -> np.ndarray:
    """
    Nettoie le masque par opérations morphologiques successives.

    - Ouverture (érosion puis dilatation) : retire les petits artefacts isolés.
    - Fermeture (dilatation puis érosion) : comble les lacunes dans les zones végétalisées.
    """
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed


def get_clean_mask(labels_2d: np.ndarray, vegetation_clusters: int | list[int]) -> np.ndarray:
    """Enchaîne construction du masque + nettoyage morphologique."""
    raw_mask = build_vegetation_mask(labels_2d, vegetation_clusters)
    return apply_morphological_cleanup(raw_mask)
