"""
Quantification de la végétation et calcul de la déforestation.
"""

import numpy as np


def compute_vegetation_surface(mask: np.ndarray) -> dict:
    """
    Calcule la surface végétalisée à partir d'un masque binaire.

    Retourne :
      - n_pixels_total     : nombre total de pixels
      - n_pixels_veg       : pixels végétation
      - surface_ratio      : fraction végétalisée [0, 1]
    """
    total = mask.size
    veg = int((mask > 0).sum())
    return {
        "n_pixels_total": total,
        "n_pixels_veg": veg,
        "surface_ratio": veg / total,
    }


def compute_deforestation(stats_t0: dict, stats_t1: dict) -> dict:
    """
    Calcule la perte de végétation entre t0 et t1.

    Formule : Perte = (surface_t0 - surface_t1) / surface_t0

    Retourne également la carte de différence pixel à pixel.
    """
    s0 = stats_t0["surface_ratio"]
    s1 = stats_t1["surface_ratio"]

    if s0 == 0:
        perte = 0.0
    else:
        perte = (s0 - s1) / s0

    return {
        "surface_t0": s0,
        "surface_t1": s1,
        "perte_relative": perte,
        "perte_pct": perte * 100,
        "gain": perte < 0,  # True si végétation a augmenté
    }


def deforestation_map(mask_t0: np.ndarray, mask_t1: np.ndarray) -> np.ndarray:
    """
    Produit une carte RGB de la déforestation pixel à pixel :
      - Rouge  : végétation perdue (présente en t0, absente en t1)
      - Vert   : végétation conservée
      - Gris   : non végétation dans les deux images
      - Bleu   : végétation gagnée (absente en t0, présente en t1)
    """
    veg0 = mask_t0 > 0
    veg1 = mask_t1 > 0

    h, w = mask_t0.shape
    carte = np.zeros((h, w, 3), dtype=np.uint8)

    carte[veg0 & veg1] = [34, 139, 34]    # vert  : conservée
    carte[veg0 & ~veg1] = [220, 50, 50]   # rouge : perdue (déforestation)
    carte[~veg0 & veg1] = [50, 100, 220]  # bleu  : gagnée (reforestation)
    carte[~veg0 & ~veg1] = [180, 180, 180]  # gris : sol/urbain

    return carte
