"""
Choix du nombre de clusters par CAH, puis segmentation K-means.
Identification automatique du cluster végétation.

Principe de cohérence inter-images :
  Le modèle K-means est entraîné sur t0, puis appliqué à t1 avec le même
  scaler et les mêmes centroïdes. Cela garantit que les clusters ont la même
  signification dans les deux images (même espace de features normalisées),
  rendant la comparaison temporelle valide.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage


def sample_pixels(feature_matrix: np.ndarray, n_samples: int = 2000, seed: int = 42) -> np.ndarray:
    """Échantillonne aléatoirement n_samples pixels pour la CAH (coût mémoire)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feature_matrix), size=min(n_samples, len(feature_matrix)), replace=False)
    return feature_matrix[idx]


def compute_linkage_matrix(sample: np.ndarray) -> np.ndarray:
    """Calcule la matrice de linkage Ward sur l'échantillon (pour affichage dendrogramme)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample)
    return linkage(scaled, method="ward")


def choose_k_from_linkage(Z: np.ndarray, max_k: int = 8) -> int:
    """
    Choisit k automatiquement en cherchant le plus grand saut de distance
    dans la matrice de linkage (méthode du saut maximal / elbow sur Ward).
    """
    last_merges = Z[-(max_k - 1):, 2]
    accelerations = np.diff(last_merges)
    k = int(accelerations.argmax()) + 2  # +2 car diff réduit d'un, et k≥2
    return k


def fit_kmeans(
    feature_matrix: np.ndarray,
    k: int,
    seed: int = 42,
) -> tuple[np.ndarray, KMeans, StandardScaler]:
    """
    Entraîne un K-means sur feature_matrix.
    Retourne les labels, le modèle et le scaler (pour réutilisation sur t1).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)  # type: ignore[call-overload]
    labels = km.fit_predict(scaled)
    return labels, km, scaler


def predict_kmeans(
    feature_matrix: np.ndarray,
    km: KMeans,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Applique un K-means déjà entraîné à une nouvelle image.
    Utilise le même scaler que lors de l'entraînement pour cohérence.
    """
    scaled = scaler.transform(feature_matrix)
    return km.predict(scaled)


def identify_vegetation_clusters(
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    green_ratio_col: int = 6,
    min_proportion: float = 0.03,
    min_coverage: float = 0.10,
) -> list[int]:
    """
    Identifie automatiquement les clusters végétation.

    Stratégie adaptative :
    1. On classe les clusters valides (>= min_proportion) par ratio vert décroissant.
    2. On sélectionne le meilleur cluster. Si sa couverture est inférieure à
       min_coverage (10% par défaut), on ajoute le cluster suivant.

    Justification : la végétation présente un ratio G/(R+G+B) supérieur au sol
    nu, aux routes et aux bâtiments. La sélection adaptative évite de prendre
    un micro-cluster outlier (< min_proportion) comme seule végétation.
    Aucune sélection manuelle n'est effectuée.
    """
    n_total = len(labels)
    n_clusters = labels.max() + 1

    candidates = []
    for c in range(n_clusters):
        mask_c = labels == c
        proportion = mask_c.sum() / n_total
        if proportion < min_proportion:
            continue
        ratio = feature_matrix[mask_c, green_ratio_col].mean()
        candidates.append((ratio, proportion, c))
    candidates.sort(reverse=True)

    selected = []
    total_coverage = 0.0
    for ratio, proportion, c in candidates:
        selected.append(c)
        total_coverage += proportion
        if total_coverage >= min_coverage:
            break

    return selected


def run_clustering_pipeline(
    feat_t0: np.ndarray,
    feat_t1: np.ndarray,
    img_shape: tuple[int, int],
    max_k: int = 8,
    n_samples_cah: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Pipeline complet cohérent sur deux images via clustering joint :
      1. CAH sur un échantillon combiné t0+t1 → choix de k
      2. K-means entraîné sur l'ensemble des pixels des deux images combinés
      3. Les labels sont séparés par image
      4. Identification du cluster végétation sur les features combinées

    Justification : en combinant t0 et t1 dans le même espace de features
    normalisé, chaque cluster représente le même type de surface dans les
    deux images (mêmes centroïdes). La comparaison temporelle est ainsi valide
    même si les images ont des luminosités différentes.
    """
    n_t0 = len(feat_t0)
    feat_combined = np.vstack([feat_t0, feat_t1])

    # CAH sur un échantillon combiné pour choisir k
    sample = sample_pixels(feat_combined, n_samples=n_samples_cah, seed=seed)
    Z = compute_linkage_matrix(sample)
    k = choose_k_from_linkage(Z, max_k=max_k)

    # K-means sur les features combinées
    labels_all, km, scaler = fit_kmeans(feat_combined, k=k, seed=seed)

    # Séparation des labels par image
    labels_t0 = labels_all[:n_t0]
    labels_t1 = labels_all[n_t0:]

    # Identification végétation sur les features combinées
    veg_clusters = identify_vegetation_clusters(labels_all, feat_combined)

    return {
        "k": k,
        "linkage_matrix": Z,
        "km_model": km,
        "scaler": scaler,
        "vegetation_clusters": veg_clusters,
        "t0": {
            "labels_2d": labels_t0.reshape(img_shape),
            "labels_1d": labels_t0,
        },
        "t1": {
            "labels_2d": labels_t1.reshape(img_shape),
            "labels_1d": labels_t1,
        },
    }
