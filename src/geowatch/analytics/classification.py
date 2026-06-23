"""Land-use and land-cover classification helpers for GeoWatch Phase 4."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geowatch.analytics.errors import ClassificationError
from geowatch.analytics.indices import compute_scene_indices, extract_canonical_bands
from geowatch.analytics.models import (
    ANALYTICS_CLASS_NAMES as _ANALYTICS_CLASS_NAMES,
)
from geowatch.analytics.models import (
    AccuracyAssessment,
    ClassificationResult,
    TransitionResult,
)
from geowatch.processing.models import RasterLayer

ANALYTICS_CLASS_NAMES = _ANALYTICS_CLASS_NAMES

CLASS_PROTOTYPE_SIGNATURES: dict[str, dict[str, float]] = {
    "Water": {
        "blue": 0.10,
        "green": 0.12,
        "red": 0.10,
        "nir": 0.02,
        "swir1": 0.01,
        "swir2": 0.01,
    },
    "Urban": {
        "blue": 0.18,
        "green": 0.20,
        "red": 0.25,
        "nir": 0.22,
        "swir1": 0.30,
        "swir2": 0.32,
    },
    "Vegetation": {
        "blue": 0.05,
        "green": 0.10,
        "red": 0.07,
        "nir": 0.60,
        "swir1": 0.20,
        "swir2": 0.15,
    },
    "Agriculture": {
        "blue": 0.07,
        "green": 0.12,
        "red": 0.10,
        "nir": 0.45,
        "swir1": 0.25,
        "swir2": 0.20,
    },
    "Bare Soil": {
        "blue": 0.20,
        "green": 0.22,
        "red": 0.24,
        "nir": 0.25,
        "swir1": 0.35,
        "swir2": 0.30,
    },
    "Forest": {
        "blue": 0.03,
        "green": 0.08,
        "red": 0.05,
        "nir": 0.75,
        "swir1": 0.18,
        "swir2": 0.10,
    },
    "Wetlands": {
        "blue": 0.08,
        "green": 0.15,
        "red": 0.10,
        "nir": 0.35,
        "swir1": 0.15,
        "swir2": 0.12,
    },
    "Snow/Ice": {
        "blue": 0.85,
        "green": 0.88,
        "red": 0.90,
        "nir": 0.70,
        "swir1": 0.40,
        "swir2": 0.30,
    },
}

DEFAULT_LULC_METHODS: tuple[str, ...] = (
    "kmeans",
    "isodata",
    "random_forest",
    "xgboost",
    "svm",
)


def classify_lulc(
    scene: RasterLayer,
    *,
    method: str,
    training_labels: NDArray[np.int64] | NDArray[np.str_] | None = None,
    index_maps: Mapping[str, NDArray[np.float32]] | None = None,
    random_state: int = 42,
) -> ClassificationResult:
    """Classify a scene using one of the supported LULC methods."""
    feature_matrix, valid_mask, feature_names, shape = _build_feature_stack(
        scene,
        index_maps=index_maps,
    )
    method_name = method.lower()
    scaled_features, scaler = _scale_features(feature_matrix)
    if method_name == "kmeans":
        labels, model_name = _classify_unsupervised(
            scaled_features,
            scaler,
            feature_names,
            cluster_strategy="kmeans",
            random_state=random_state,
        )
    elif method_name == "isodata":
        labels, model_name = _classify_unsupervised(
            scaled_features,
            scaler,
            feature_names,
            cluster_strategy="isodata",
            random_state=random_state,
        )
    elif method_name in {"random_forest", "xgboost", "svm"}:
        if training_labels is None:
            message = f"{method_name} classification requires training labels."
            logger.error(message)
            raise ClassificationError(message)
        labels, model_name = _classify_supervised(
            scaled_features,
            valid_mask,
            shape,
            training_labels,
            method=method_name,
            random_state=random_state,
        )
    else:
        message = f"Unsupported LULC method: {method}"
        logger.error(message)
        raise ClassificationError(message)

    classification_map = np.full(valid_mask.size, -1, dtype=np.int64)
    classification_map[valid_mask] = labels
    label_grid = classification_map.reshape(shape)
    counts = _count_labels(label_grid, ANALYTICS_CLASS_NAMES)
    logger.info("Classified scene {} with method {}", scene.name, method_name)
    return ClassificationResult(
        method=method_name,
        labels=label_grid,
        class_names=ANALYTICS_CLASS_NAMES,
        counts=counts,
        model_name=model_name,
        feature_names=feature_names,
        metadata={
            "random_state": random_state,
            "valid_pixels": int(valid_mask.sum()),
        },
    )


def build_transition_result(
    classification_t1: ClassificationResult,
    classification_t2: ClassificationResult,
) -> TransitionResult:
    """Build transition and change matrices from two LULC maps."""
    if classification_t1.labels.shape != classification_t2.labels.shape:
        message = "Classification maps must have matching shapes."
        logger.error(message)
        raise ClassificationError(message)
    class_names = classification_t1.class_names
    transition_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    valid = (classification_t1.labels >= 0) & (classification_t2.labels >= 0)
    flat_t1 = classification_t1.labels[valid].reshape(-1)
    flat_t2 = classification_t2.labels[valid].reshape(-1)
    for from_label, to_label in zip(flat_t1, flat_t2, strict=False):
        transition_matrix[int(from_label), int(to_label)] += 1
    change_matrix = transition_matrix.copy()
    np.fill_diagonal(change_matrix, 0)
    changed_pixels = int(change_matrix.sum())
    logger.info("Built transition matrix with {} changed pixels", changed_pixels)
    return TransitionResult(
        class_names=class_names,
        transition_matrix=transition_matrix,
        change_matrix=change_matrix,
        changed_pixels=changed_pixels,
    )


def assess_accuracy(
    reference_labels: NDArray[np.int_] | NDArray[np.str_],
    predicted_labels: NDArray[np.int64],
    *,
    class_names: Sequence[str] = ANALYTICS_CLASS_NAMES,
) -> AccuracyAssessment:
    """Compare predicted labels against a reference map."""
    reference_encoded = _encode_labels(reference_labels, class_names)
    predicted = cast(NDArray[np.int64], np.asarray(predicted_labels, dtype=np.int64))
    if reference_encoded.shape != predicted.shape:
        message = "Reference and predicted labels must share the same shape."
        logger.error(message)
        raise ClassificationError(message)
    valid = (reference_encoded >= 0) & (predicted >= 0)
    if not np.any(valid):
        message = "Accuracy assessment requires at least one valid pixel."
        logger.error(message)
        raise ClassificationError(message)
    reference_flat = reference_encoded[valid].reshape(-1)
    predicted_flat = predicted[valid].reshape(-1)
    label_indices = np.arange(len(class_names), dtype=np.int64)
    matrix = cast(
        NDArray[np.int64],
        confusion_matrix(reference_flat, predicted_flat, labels=label_indices),
    )
    overall_accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    kappa = float(
        cohen_kappa_score(reference_flat, predicted_flat, labels=label_indices)
    )
    per_class_accuracy: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        row_total = matrix[index].sum()
        per_class_accuracy[class_name] = (
            float(matrix[index, index] / row_total) if row_total else 0.0
        )
    logger.info(
        "Computed accuracy assessment with overall accuracy {:.2%}",
        overall_accuracy,
    )
    return AccuracyAssessment(
        class_names=tuple(class_names),
        confusion_matrix=matrix,
        overall_accuracy=overall_accuracy,
        kappa=kappa,
        per_class_accuracy=per_class_accuracy,
    )


def _build_feature_stack(
    scene: RasterLayer,
    *,
    index_maps: Mapping[str, NDArray[np.float32]] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[str, ...], tuple[int, int]]:
    """Construct a feature matrix for LULC classification."""
    bands = extract_canonical_bands(scene)
    index_bundle = dict(index_maps or compute_scene_indices(scene))
    feature_names = (
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        *index_bundle.keys(),
    )
    feature_stack = np.stack(
        [
            bands["blue"],
            bands["green"],
            bands["red"],
            bands["nir"],
            bands["swir1"],
            bands["swir2"],
            *[index_bundle[name] for name in index_bundle],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    valid_mask = np.all(np.isfinite(feature_stack), axis=0)
    feature_matrix = feature_stack.reshape(feature_stack.shape[0], -1).T
    return (
        feature_matrix[valid_mask.reshape(-1)],
        valid_mask.reshape(-1),
        tuple(feature_names),
        (scene.grid.height, scene.grid.width),
    )


def _scale_features(
    features: NDArray[np.float32],
) -> tuple[NDArray[np.float32], StandardScaler]:
    """Scale features for clustering and supervised classification."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled.astype(np.float32, copy=False), scaler


def _classify_unsupervised(
    features: NDArray[np.float32],
    scaler: StandardScaler,
    feature_names: tuple[str, ...],
    *,
    cluster_strategy: str,
    random_state: int,
) -> tuple[NDArray[np.int64], str]:
    """Classify a feature stack using an unsupervised clustering strategy."""
    if cluster_strategy == "kmeans":
        labels, centers = _run_kmeans(features, random_state=random_state)
        model_name = "kmeans"
    elif cluster_strategy == "isodata":
        labels, centers = _run_isodata(features, random_state=random_state)
        model_name = "isodata"
    else:
        raise ClassificationError(
            f"Unsupported unsupervised strategy: {cluster_strategy}"
        )
    prototype_matrix = _prototype_matrix(feature_names)
    scaled_prototypes = scaler.transform(prototype_matrix)
    cluster_to_class = _map_clusters_to_classes(centers, scaled_prototypes)
    mapped = np.asarray(
        [cluster_to_class[int(label)] for label in labels],
        dtype=np.int64,
    )
    return mapped, model_name


def _classify_supervised(
    features: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
    shape: tuple[int, int],
    training_labels: NDArray[np.int64] | NDArray[np.str_] | None,
    *,
    method: str,
    random_state: int,
) -> tuple[NDArray[np.int64], str]:
    """Classify a feature stack using a supervised estimator."""
    if training_labels is None:
        message = "Supervised classification requires training labels."
        logger.error(message)
        raise ClassificationError(message)
    encoded_labels = _encode_labels(training_labels, ANALYTICS_CLASS_NAMES)
    if encoded_labels.shape != shape:
        message = "Training labels must match the raster shape."
        logger.error(message)
        raise ClassificationError(message)
    y = encoded_labels.reshape(-1)[valid_mask]
    if np.unique(y).size < 2:
        raise ClassificationError(
            "Supervised classification requires multiple classes."
        )
    x = features
    if method == "random_forest":
        estimator: Any = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        )
        model_name = "random_forest"
    elif method == "svm":
        estimator = SVC(
            kernel="rbf", class_weight="balanced", random_state=random_state
        )
        model_name = "svm_rbf"
    else:
        estimator = _build_xgboost_estimator(random_state=random_state)
        model_name = estimator.__class__.__name__
    estimator.fit(x, y)
    predicted = cast(
        NDArray[np.int64], np.asarray(estimator.predict(x), dtype=np.int64)
    )
    return predicted, model_name


def _run_kmeans(
    features: NDArray[np.float32],
    *,
    random_state: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Fit KMeans using the canonical land-cover class count."""
    cluster_count = min(len(ANALYTICS_CLASS_NAMES), features.shape[0])
    model = KMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(features)
    return labels.astype(np.int64, copy=False), model.cluster_centers_.astype(
        np.float32,
        copy=False,
    )


def _run_isodata(
    features: NDArray[np.float32],
    *,
    random_state: int,
    max_iterations: int = 5,
    split_std: float = 0.75,
    merge_distance: float = 0.85,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Run a simplified ISODATA clustering routine."""
    labels, centers = _run_kmeans(features, random_state=random_state)
    current_centers = centers
    for _iteration in range(max_iterations):
        clusters: list[NDArray[np.float32]] = []
        for cluster_index in range(current_centers.shape[0]):
            cluster_features = features[labels == cluster_index]
            if cluster_features.size == 0:
                continue
            cluster_std = np.std(cluster_features, axis=0)
            should_split = (
                cluster_features.shape[0] >= 2
                and float(np.max(cluster_std)) > split_std
                and current_centers.shape[0] < len(ANALYTICS_CLASS_NAMES)
            )
            if should_split:
                offset = np.zeros_like(current_centers[cluster_index])
                dominant_axis = int(np.argmax(cluster_std))
                offset[dominant_axis] = cluster_std[dominant_axis] / 2.0
                clusters.append(current_centers[cluster_index] - offset)
                clusters.append(current_centers[cluster_index] + offset)
            else:
                clusters.append(current_centers[cluster_index])
        merged = _merge_centers(np.asarray(clusters, dtype=np.float32), merge_distance)
        if merged.shape[0] == current_centers.shape[0]:
            break
        model = KMeans(
            n_clusters=min(merged.shape[0], features.shape[0]),
            init=merged,
            n_init=1,
            random_state=random_state,
        )
        labels = model.fit_predict(features)
        current_centers = model.cluster_centers_.astype(np.float32, copy=False)
    return labels.astype(np.int64, copy=False), current_centers


def _merge_centers(
    centers: NDArray[np.float32],
    merge_distance: float,
) -> NDArray[np.float32]:
    """Merge cluster centers that are very close together."""
    if centers.shape[0] <= 1:
        return centers
    remaining = list(range(centers.shape[0]))
    merged: list[NDArray[np.float32]] = []
    while remaining:
        index = remaining.pop(0)
        current = centers[index]
        close = [
            other
            for other in remaining
            if np.linalg.norm(current - centers[other]) < merge_distance
        ]
        if close:
            cluster_indices = [index, *close]
            remaining = [other for other in remaining if other not in close]
            merged.append(np.mean(centers[cluster_indices], axis=0))
        else:
            merged.append(current)
    return np.asarray(merged, dtype=np.float32)


def _prototype_matrix(feature_names: tuple[str, ...]) -> NDArray[np.float32]:
    """Build class prototypes in the requested feature space."""
    prototypes = []
    for class_name in ANALYTICS_CLASS_NAMES:
        reflectance = CLASS_PROTOTYPE_SIGNATURES[class_name]
        prototypes.append(_feature_vector_from_reflectance(reflectance, feature_names))
    return np.asarray(prototypes, dtype=np.float32)


def _feature_vector_from_reflectance(
    reflectance: Mapping[str, float],
    feature_names: tuple[str, ...],
) -> NDArray[np.float32]:
    """Construct a feature vector for a single land-cover prototype."""
    bands = {
        "blue": float(reflectance["blue"]),
        "green": float(reflectance["green"]),
        "red": float(reflectance["red"]),
        "nir": float(reflectance["nir"]),
        "swir1": float(reflectance["swir1"]),
        "swir2": float(reflectance["swir2"]),
    }
    index_maps = compute_scene_indices(_synthetic_scene_from_reflectance(bands))
    values: list[float] = []
    for feature_name in feature_names:
        if feature_name in bands:
            values.append(bands[feature_name])
        else:
            values.append(float(index_maps[feature_name][0, 0]))
    return np.asarray(values, dtype=np.float32)


def _synthetic_scene_from_reflectance(reflectance: Mapping[str, float]) -> RasterLayer:
    """Construct a one-pixel synthetic scene for prototype generation."""
    data = np.asarray(
        [
            [[reflectance["blue"]]],
            [[reflectance["green"]]],
            [[reflectance["red"]]],
            [[reflectance["nir"]]],
            [[reflectance["swir1"]]],
            [[reflectance["swir2"]]],
        ],
        dtype=np.float32,
    )
    from geowatch.processing.models import RasterGrid

    grid = RasterGrid(
        crs="EPSG:4326",
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        width=1,
        height=1,
        band_names=("blue", "green", "red", "nir", "swir1", "swir2"),
        nodata=None,
    )
    return RasterLayer(name="prototype", data=data, grid=grid)


def _map_clusters_to_classes(
    cluster_centers: NDArray[np.float32],
    prototype_vectors: NDArray[np.float32],
) -> dict[int, int]:
    """Assign cluster centers to the nearest land-cover classes."""
    distances = np.linalg.norm(
        cluster_centers[:, None, :] - prototype_vectors[None, :, :],
        axis=2,
    )
    cluster_indices, class_indices = linear_sum_assignment(distances)
    assignment = {
        int(cluster): int(cls)
        for cluster, cls in zip(cluster_indices, class_indices, strict=False)
    }
    if len(assignment) < cluster_centers.shape[0]:
        for cluster_index in range(cluster_centers.shape[0]):
            if cluster_index not in assignment:
                assignment[cluster_index] = int(np.argmin(distances[cluster_index]))
    return assignment


def _build_xgboost_estimator(random_state: int) -> Any:
    """Return an XGBoost estimator or a gradient boosting fallback."""
    if importlib.util.find_spec("xgboost") is not None:
        xgb = cast(Any, importlib.import_module("xgboost"))
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        )
    logger.warning("xgboost is unavailable; using GradientBoostingClassifier fallback.")
    return GradientBoostingClassifier(random_state=random_state)


def _encode_labels(
    labels: NDArray[np.int_] | NDArray[np.str_],
    class_names: Sequence[str],
) -> NDArray[np.int64]:
    """Encode string or integer labels into the canonical class ordering."""
    array = np.asarray(labels)
    if array.dtype.kind in {"U", "S", "O"}:
        mapping = {name: index for index, name in enumerate(class_names)}
        flat: list[int] = []
        for value in array.reshape(-1):
            label = str(value)
            if label not in mapping:
                raise ClassificationError(f"Unknown land-cover label: {label}")
            flat.append(mapping[label])
        return cast(
            NDArray[np.int64],
            np.asarray(flat, dtype=np.int64).reshape(array.shape),
        )
    encoded = cast(NDArray[np.int64], array.astype(np.int64, copy=False))
    if np.any((encoded < 0) | (encoded >= len(class_names))):
        raise ClassificationError(
            "Encoded labels must fall within the class index range."
        )
    return encoded


def _count_labels(
    labels: NDArray[np.int64],
    class_names: Sequence[str],
) -> dict[str, int]:
    """Count class assignments in a label grid."""
    counts: dict[str, int] = {}
    for index, class_name in enumerate(class_names):
        counts[class_name] = int(np.sum(labels == index))
    return counts
