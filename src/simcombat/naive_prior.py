"""Legacy-compatible similarity-weighted moment priors."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata

from .similarity import normalize_similarity_weights


def _similarity_weights(
    standardized: np.ndarray,
    batches: list[np.ndarray],
    methods: list[str],
    options: dict[str, object],
    supplied_similarity: np.ndarray | None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    n_batches, n_features = len(batches), standardized.shape[0]
    eps = 1e-50
    method_weights = options.get("method_weights")
    if method_weights is None:
        method_weights = {method: 1.0 / len(methods) for method in methods}
    else:
        method_weights = dict(method_weights)
        total = float(sum(method_weights.values()))
        if total <= 0 or any(float(value) < 0 for value in method_weights.values()):
            raise ValueError("method_weights must be nonnegative with a positive sum.")
        method_weights = {key: float(value) / total for key, value in method_weights.items()}

    coords = options.get("feature_coords")
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] != n_features or not np.all(np.isfinite(coords)):
            raise ValueError("feature_coords must be finite with one row per feature.")
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        bandwidth = float(options.get("sigma", np.std(distances)))
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 1.0
        spatial = np.exp(-(distances**2) / (2.0 * bandwidth**2))
    else:
        spatial = np.ones((n_features, n_features))

    if supplied_similarity is not None:
        custom = np.asarray(supplied_similarity, dtype=float)
        if custom.shape != (n_features, n_features) or not np.all(np.isfinite(custom)):
            raise ValueError("similarity must be a finite square feature-by-feature matrix.")
        if np.any(custom < 0):
            raise ValueError("similarity entries must be nonnegative.")
    elif options.get("similarity_matrix") is not None:
        custom = np.asarray(options["similarity_matrix"], dtype=float)
        if custom.shape != (n_features, n_features) or not np.all(np.isfinite(custom)):
            raise ValueError("similarity_matrix must be finite and feature-by-feature.")
        if np.any(custom < 0):
            raise ValueError("similarity_matrix entries must be nonnegative.")
    else:
        custom = None

    floor = max(float(options.get("similarity_floor", 0.01)), eps)
    normalize = bool(options.get("normalize", True))
    weights = np.zeros((n_batches, n_features, n_features), dtype=float)
    warning_messages: list[str] = []
    for batch_index, sample_indices in enumerate(batches):
        n_subjects = len(sample_indices)
        if n_subjects >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = np.corrcoef(standardized[:, sample_indices])
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            corr_similarity = (1.0 + corr) / 2.0 + eps
            np.fill_diagonal(corr_similarity, 1.0)
            variance = np.var(standardized[:, sample_indices], axis=1, ddof=1)
            if n_subjects < int(options.get("corr_min_n", 50)) and any(
                name in methods for name in ("correlation_similarity", "covariance_similarity")
            ):
                warning_messages.append(
                    f"Batch {batch_index} has n={n_subjects}; correlation similarity may be unstable."
                )
        else:
            corr_similarity = np.eye(n_features)
            variance = np.zeros(n_features)

        variance_difference = np.abs(variance[:, None] - variance[None, :])
        max_difference = float(np.max(variance_difference))
        variance_similarity = (
            np.ones_like(variance_difference)
            if max_difference <= 0
            else np.clip(1.0 - variance_difference / (max_difference + eps), 0.0, 1.0)
        )
        means = (
            np.mean(standardized[:, sample_indices], axis=1)
            if n_subjects
            else np.zeros(n_features)
        )
        ranks = rankdata(means, method="average")
        rank_distance = np.abs(ranks[:, None] - ranks[None, :])
        rank_similarity = 1.0 - rank_distance / max(n_features - 1, 1)

        combined = np.zeros((n_features, n_features), dtype=float)
        for method in methods:
            weight = float(method_weights.get(method, 0.0))
            if method == "correlation_similarity":
                matrix = corr_similarity
            elif method == "covariance_similarity":
                matrix = np.abs(np.cov(standardized[:, sample_indices], ddof=1)) if n_subjects > 1 else np.eye(n_features)
                matrix = matrix / max(float(np.max(matrix)), eps)
            elif method == "variance_similarity":
                matrix = variance_similarity
            elif method in {"magnitude_similarity", "directional_bias"}:
                matrix = rank_similarity
            elif method == "spatial_proximity":
                matrix = spatial
            elif method in {"custom", "similarity_matrix"}:
                matrix = custom if custom is not None else np.zeros_like(combined)
            else:
                raise ValueError(f"Unsupported naive similarity method: {method!r}.")
            matrix = np.maximum(np.nan_to_num(matrix), 0.0)
            matrix = np.maximum(matrix, floor)
            if normalize:
                matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), eps)
            combined += weight * matrix
        if normalize:
            combined /= np.maximum(combined.sum(axis=1, keepdims=True), eps)
        normalized, _ = normalize_similarity_weights(
            combined,
            trust=float(options.get("trust", 1.0)),
            exclude_diagonal=bool(options.get("exclude_diagonal", False)),
        )
        weights[batch_index] = normalized
    return weights, warning_messages, method_weights


def construct_naive_priors(
    standardized: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batches: list[np.ndarray],
    global_priors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    methods: list[str] | None = None,
    options: dict[str, object] | None = None,
    similarity: np.ndarray | None = None,
    legacy_compat: bool = True,
) -> dict[str, object]:
    """Construct weighted-moment priors, optionally preserving legacy controls.

    ``legacy_compat=True`` retains the original 0.01 similarity floor, ESS
    fallback threshold of five, and per-method row normalization unless those
    values are explicitly overridden in ``options``.
    """
    options = {} if options is None else dict(options)
    if legacy_compat:
        options.setdefault("similarity_floor", 0.01)
        options.setdefault("min_effective", 5)
        options.setdefault("normalize", True)
    methods = ["correlation_similarity"] if methods is None else list(methods)
    if not methods:
        raise ValueError("At least one naive similarity method is required.")
    weights, warning_messages, method_weights = _similarity_weights(
        standardized, batches, methods, options, similarity
    )
    gamma_global, variance_global, shape_global, scale_global = global_priors
    n_batches, n_features = gamma_hat.shape
    minimum_effective = int(options.get("min_effective", 5))
    if minimum_effective < 1:
        raise ValueError("min_effective must be at least one.")
    gamma_mean_local = np.zeros((n_batches, n_features))
    gamma_variance_local = np.zeros_like(gamma_mean_local)
    delta_shape_local = np.zeros_like(gamma_mean_local)
    delta_scale_local = np.zeros_like(gamma_mean_local)
    effective_size = np.zeros_like(gamma_mean_local)
    fallback = np.zeros((n_batches, n_features), dtype=bool)

    for batch_index in range(n_batches):
        for feature_index in range(n_features):
            row = weights[batch_index, feature_index].copy()
            row_sum = float(row.sum())
            if row_sum <= 0:
                row[:] = 1.0 / n_features
            else:
                row /= row_sum
            ess = 1.0 / float(np.sum(row**2))
            effective_size[batch_index, feature_index] = ess
            if ess < minimum_effective:
                gamma_mean_local[batch_index, feature_index] = gamma_global[batch_index]
                gamma_variance_local[batch_index, feature_index] = variance_global[batch_index]
                delta_shape_local[batch_index, feature_index] = shape_global[batch_index]
                delta_scale_local[batch_index, feature_index] = scale_global[batch_index]
                fallback[batch_index, feature_index] = True
                continue

            gamma_values = gamma_hat[batch_index]
            delta_values = delta_hat[batch_index]
            gamma_mean = float(np.sum(row * gamma_values))
            correction = max(1.0 - float(np.sum(row**2)), np.finfo(float).eps)
            gamma_variance = float(np.sum(row * (gamma_values - gamma_mean) ** 2) / correction)
            delta_mean = float(np.sum(row * delta_values))
            delta_variance = float(np.sum(row * (delta_values - delta_mean) ** 2) / correction)
            if (
                gamma_variance <= np.finfo(float).eps
                or delta_variance <= np.finfo(float).eps * max(1.0, delta_mean**2)
                or delta_mean <= 0
            ):
                gamma_mean_local[batch_index, feature_index] = gamma_global[batch_index]
                gamma_variance_local[batch_index, feature_index] = variance_global[batch_index]
                delta_shape_local[batch_index, feature_index] = shape_global[batch_index]
                delta_scale_local[batch_index, feature_index] = scale_global[batch_index]
                fallback[batch_index, feature_index] = True
                continue
            gamma_mean_local[batch_index, feature_index] = gamma_mean
            gamma_variance_local[batch_index, feature_index] = gamma_variance
            delta_shape_local[batch_index, feature_index] = (
                2.0 * delta_variance + delta_mean**2
            ) / delta_variance
            delta_scale_local[batch_index, feature_index] = (
                delta_mean * delta_variance + delta_mean**3
            ) / delta_variance

    return {
        "gamma_mean": gamma_mean_local,
        "gamma_variance": gamma_variance_local,
        "delta_shape": delta_shape_local,
        "delta_scale": delta_scale_local,
        "weights": weights,
        "effective_sample_size": effective_size,
        "used_global_fallback": fallback,
        "warnings": warning_messages,
        "weight_methods": methods,
        "method_weights": method_weights,
        "legacy_compat": bool(legacy_compat),
    }
