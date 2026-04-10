# Projet de traitement d'image — détection de déforestation

Ce projet implémente un pipeline modulaire de traitement d'images pour détecter et quantifier la déforestation entre deux images satellites.

Structure du dépôt :

- `src/` : code métier Python
- `data/` : emplacement prévu pour les images d'entrée `t0` et `t1`
- `notebooks/` : démonstration et visualisation
- `outputs/` : résultats générés

Fichiers principaux :

- `src/forest_pipeline.py` : fonctions de prétraitement, extraction de features, clustering, segmentation, identification automatique de la végétation et quantification
- `notebooks/Projet_demo.ipynb` : démonstration d’utilisation du pipeline avec un exemple de données

Installation :

```bash
pip install -r requirements.txt
```

Utilisation :

```bash
python -c "from src.forest_pipeline import load_image, segment_kmeans; print('OK')"
```

Données :

Déposez vos images satellite `t0` et `t1` dans `data/`.
