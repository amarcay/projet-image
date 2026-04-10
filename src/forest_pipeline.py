import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from skimage import color, io, morphology, util
from matplotlib import cm


def load_image(path, as_float=True):
    image = io.imread(path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = util.img_as_float(image)
    return image


def rgb_to_hsv(image):
    return color.rgb2hsv(image)


def green_ratio(image, eps=1e-8):
    rgb = image[..., :3]
    total = np.sum(rgb, axis=-1, keepdims=True) + eps
    ratio = rgb[..., 1:2] / total
    return np.clip(ratio[..., 0], 0.0, 1.0)


def local_mean_variance(gray, radius=7):
    if gray.ndim != 2:
        raise ValueError('Local features require a grayscale image')
    size = 2 * radius + 1
    mean = ndi.uniform_filter(gray, size=size)
    mean_sq = ndi.uniform_filter(gray**2, size=size)
    variance = np.clip(mean_sq - mean**2, 0.0, None)
    return mean, variance


def build_pixel_features(image, include_local=True, radius=7):
    rgb = image[..., :3]
    hsv = rgb_to_hsv(rgb)
    ratio = green_ratio(rgb)[..., np.newaxis]
    gray = color.rgb2gray(rgb)
    features = [rgb, hsv, ratio]
    if include_local:
        local_mean, local_var = local_mean_variance(gray, radius=radius)
        features.append(local_mean[..., np.newaxis])
        features.append(local_var[..., np.newaxis])
    return np.concatenate(features, axis=-1)


def sample_pixel_features(features, n_samples=2500, random_state=0):
    height, width, channels = features.shape
    pixels = features.reshape(-1, channels)
    if n_samples >= len(pixels):
        return pixels
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(pixels), size=n_samples, replace=False)
    return pixels[indices]


def choose_cluster_count(sampled_features, k_min=2, k_max=6):
    scores = {}
    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(sampled_features)
        if len(np.unique(labels)) == 1:
            scores[k] = np.nan
            continue
        try:
            score = silhouette_score(sampled_features, labels)
        except Exception:
            score = np.nan
        scores[k] = float(score)
    return scores


def hierarchical_clustering(sampled_features, n_clusters=4, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(sampled_features)


def segment_kmeans(image, n_clusters=4, random_state=0, include_local=True, radius=7):
    features = build_pixel_features(image, include_local=include_local, radius=radius)
    height, width, channels = features.shape
    pixel_features = features.reshape(-1, channels)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(pixel_features)
    return labels.reshape(height, width), model


def identify_vegetation_cluster(labels, image, n_clusters):
    ratios = green_ratio(image)
    cluster_scores = []
    for cluster in range(n_clusters):
        mask = labels == cluster
        if mask.sum() == 0:
            cluster_scores.append(0.0)
        else:
            cluster_scores.append(float(ratios[mask].mean()))
    vegetation_cluster = int(np.argmax(cluster_scores))
    vegetation_mask = labels == vegetation_cluster
    return vegetation_mask, vegetation_cluster, cluster_scores


def clean_mask(mask, opening_radius=3, closing_radius=3, min_size=500):
    cleaned = morphology.binary_opening(mask, morphology.disk(opening_radius))
    cleaned = morphology.binary_closing(cleaned, morphology.disk(closing_radius))
    cleaned = morphology.remove_small_objects(cleaned, min_size=min_size)
    return cleaned


def compute_area(mask, pixel_area=1.0):
    return float(mask.sum() * pixel_area)


def deforestation_map(mask_before, mask_after):
    lost = np.logical_and(mask_before, np.logical_not(mask_after))
    gained = np.logical_and(np.logical_not(mask_before), mask_after)
    unchanged = np.logical_and(mask_before, mask_after)
    return {'lost': lost, 'gained': gained, 'unchanged': unchanged}


def overlay_mask(image, mask, color_rgb=(1.0, 0.0, 0.0), alpha=0.4):
    overlay = np.copy(image)
    if overlay.dtype != np.float32 and overlay.dtype != np.float64:
        overlay = util.img_as_float(overlay)
    color_layer = np.zeros_like(overlay)
    color_layer[..., 0] = color_rgb[0]
    color_layer[..., 1] = color_rgb[1]
    color_layer[..., 2] = color_rgb[2]
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_layer[mask]
    return overlay


def color_segments(labels, colormap='tab10'):
    cmap = cm.get_cmap(colormap)
    n_clusters = int(labels.max() + 1)
    color_image = cmap(labels / max(n_clusters - 1, 1))
    return color_image[..., :3]
