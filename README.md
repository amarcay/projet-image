# Détection de déforestation par traitement d'image satellite

Projet de fin de module — Traitement d'image  
Pipeline non supervisé de détection et quantification de la déforestation à partir de paires d'images satellites.

---

## Objectif

Concevoir un pipeline cohérent permettant de :
- identifier les zones végétalisées dans deux images d'une même zone à des dates différentes
- comparer les couvertures végétales (t0 → t1)
- quantifier la déforestation et produire une carte de changement

---

## Structure du projet

```
projet/
├── data/
│   ├── Lot1_A/          # Images AVANT (t0)
│   │   ├── val_3.png
│   │   └── train_128.png
│   └── Lot1_B/          # Images APRÈS (t1)
│       ├── val_3.png
│       └── train_128.png
├── src/                 # Modules Python
│   ├── preprocessing.py     # Chargement, CLAHE, histogram matching
│   ├── features.py          # Construction des features par pixel
│   ├── clustering.py        # CAH (choix de k) + K-means joint
│   ├── postprocessing.py    # Opérations morphologiques
│   ├── quantification.py    # Surface végétale, perte, carte de diff
│   └── visualization.py     # Histogrammes, segmentation, cartes
├── notebooks/
│   └── pipeline.ipynb   # Démonstration complète du pipeline
├── outputs/             # Résultats générés (cartes de déforestation)
└── pyproject.toml
```

---

## Pipeline

### 1. Prétraitement
- **CLAHE** sur le canal L (espace LAB) : normalise le contraste local sans altérer les couleurs
- **Histogram matching** : aligne la distribution des couleurs de t1 sur t0 pour un clustering cohérent

### 2. Features par pixel (10 dimensions)

| # | Feature | Rôle |
|---|---|---|
| 0–2 | R, G, B normalisés | Couleur brute |
| 3–5 | H, S, V | Teinte stable face aux variations de luminosité |
| 6 | Ratio vert `G/(R+G+B)` | Indice de végétation, invariant à la luminosité |
| 7 | Moyenne locale (5×5) | Contexte spatial |
| 8 | Variance locale (5×5) | Texture rugueuse = végétation |
| 9 | Magnitude Sobel | Richesse en contours (feuilles, branches) |

### 3. CAH → Choix de k
Classification Ascendante Hiérarchique (Ward) sur 2 000 pixels échantillonnés depuis les deux images combinées. Le k est choisi au saut maximal de distance dans le dendrogramme.

### 4. Segmentation K-means joint
K-means entraîné sur les pixels de t0 et t1 **simultanément** : chaque cluster représente le même type de surface dans les deux images, garantissant une comparaison temporelle valide.

### 5. Identification automatique de la végétation
Le cluster végétation est celui dont le ratio vert moyen est le plus élevé (parmi les clusters couvrant ≥ 3 % de l'image). Si la couverture est insuffisante (< 10 %), les clusters suivants sont ajoutés. Aucune sélection manuelle.

### 6. Post-traitement morphologique
- **Ouverture** (noyau 3×3) : supprime les artefacts isolés
- **Fermeture** (noyau 5×5) : comble les trous dans les zones végétalisées

### 7. Quantification

$$\text{Perte} = \frac{\text{surface}_{t0} - \text{surface}_{t1}}{\text{surface}_{t0}}$$

---

## Résultats

| Paire | Surface t0 | Surface t1 | Perte |
|---|---|---|---|
| val_3 | 16.7 % | 9.6 % | **−42.7 %** |
| train_128 | 25.0 % | 33.2 % | +32 % (pelouses confondues avec forêt) |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
```

## Utilisation

```bash
cd notebooks
jupyter notebook pipeline.ipynb
```

Pour analyser une nouvelle paire d'images, modifier uniquement le bloc `IMAGE_PAIRS` dans la cellule de configuration du notebook :

```python
IMAGE_PAIRS = [
    {
        "name": "mon_lot",
        "t0":   "../data/MonLot/avant.png",
        "t1":   "../data/MonLot/apres.png",
    },
]
```

---

## Dépendances

| Librairie | Usage |
|---|---|
| `opencv-python` | Chargement, CLAHE, morphologie |
| `numpy` | Calculs matriciels |
| `scikit-learn` | K-means, StandardScaler |
| `scikit-image` | Histogram matching |
| `scipy` | CAH (linkage Ward) |
| `matplotlib` | Visualisations |

---

## Limites

- La méthode ne distingue pas forêt, pelouse et arbustes — tout pixel vert est classé végétation.
- L'histogram matching suppose des conditions photométriques comparables entre les deux images.
- Sans géoréférencement, les surfaces en pixels ne correspondent pas à des hectares réels.
