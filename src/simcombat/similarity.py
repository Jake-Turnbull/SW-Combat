"""Validated structural feature similarities and normalized neighborhood weights."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _validate_matrix(
    matrix: np.ndarray,
    n_features: int,
    name: str,
    *,
    nonnegative: bool = True,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (n_features, n_features):
        raise ValueError(f"{name} must have shape ({n_features}, {n_features}).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(values, values.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} must be symmetric.")
    if nonnegative and np.any(values < 0):
        raise ValueError(f"{name} entries must be nonnegative.")
    return values


def normalize_similarity_weights(
    affinity: np.ndarray,
    *,
    trust: float = 1.0,
    exclude_diagonal: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Normalize a finite nonnegative affinity matrix into row weights.

    Unlike raw similarity sources, row-wise affinities need not be symmetric.
    """
    values = np.asarray(affinity, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] < 1:
        raise ValueError("affinity must be a nonempty square matrix.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("affinity must contain finite nonnegative values.")
    if not np.isfinite(trust) or not 0.0 <= trust <= 1.0:
        raise ValueError("trust must lie in [0, 1].")
    n_features = values.shape[0]
    allowed = np.ones_like(values, dtype=bool)
    if exclude_diagonal:
        if n_features < 2:
            raise ValueError("Cannot exclude the diagonal with fewer than two features.")
        np.fill_diagonal(allowed, False)
    values = np.where(allowed, values, 0.0)
    raw_sums = values.sum(axis=1)
    normalized = np.zeros_like(values)
    fallback = raw_sums <= 0
    for row_index, row_sum in enumerate(raw_sums):
        if row_sum > 0:
            normalized[row_index] = values[row_index] / row_sum
        else:
            normalized[row_index, allowed[row_index]] = 1.0 / np.sum(allowed[row_index])
    uniform = allowed.astype(float)
    uniform /= uniform.sum(axis=1, keepdims=True)
    weights = trust * normalized + (1.0 - trust) * uniform
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, {
        "structural_row_sums": raw_sums,
        "uniform_fallback_rows": fallback,
        "effective_neighborhood_size": 1.0 / np.sum(weights**2, axis=1),
        "trust": float(trust),
        "exclude_diagonal": bool(exclude_diagonal),
    }


def build_similarity(
    n_features: int,
    *,
    feature_coords: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    feature_families: object | None = None,
    overlap: np.ndarray | None = None,
    custom_similarity: np.ndarray | None = None,
    source_weights: Mapping[str, float] | None = None,
    correlation_mode: str = "absolute",
    bandwidth_mm: float | None = None,
    trust: float = 1.0,
    exclude_diagonal: bool = False,
    similarity_floor: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build row-normalized feature-neighborhood weights.

    Correlations are explicitly converted to nonnegative similarity using either
    ``absolute`` (``abs(r)``) or ``signed_shifted`` (``(r + 1) / 2``). Coordinates
    are three-dimensional millimetres and the Gaussian kernel uses Euclidean
    distance in all three dimensions. Structural weights are blended with uniform
    weights as ``trust * structural + (1 - trust) * uniform`` before
    normalization. An all-zero structural row falls back to uniform neighbors.
    """
    if not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    if not 0.0 <= trust <= 1.0:
        raise ValueError("trust must lie in [0, 1].")
    if not np.isfinite(similarity_floor) or similarity_floor < 0:
        raise ValueError("similarity_floor must be finite and nonnegative.")
    if correlation_mode not in {"absolute", "signed_shifted"}:
        raise ValueError("correlation_mode must be 'absolute' or 'signed_shifted'.")

    n_features = int(n_features)
    sources: dict[str, np.ndarray] = {}
    resolved_bandwidth: float | None = None
    if feature_coords is not None:
        coords = np.asarray(feature_coords, dtype=float)
        if coords.shape != (n_features, 3) or not np.all(np.isfinite(coords)):
            raise ValueError("feature_coords must be finite xyz coordinates in millimetres.")
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        if bandwidth_mm is None:
            positive_distances = distances[distances > 0]
            resolved_bandwidth = (
                float(np.median(positive_distances)) if positive_distances.size else 1.0
            )
        else:
            resolved_bandwidth = float(bandwidth_mm)
        if not np.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0:
            raise ValueError("bandwidth_mm must be positive and finite.")
        sources["spatial"] = np.exp(
            -(distances**2) / (2.0 * resolved_bandwidth**2)
        )

    if residuals is not None:
        values = np.asarray(residuals, dtype=float)
        if values.ndim != 2 or values.shape[0] != n_features:
            raise ValueError("residuals must have shape (features, subjects).")
        if values.shape[1] < 2 or not np.all(np.isfinite(values)):
            raise ValueError("residuals require at least two finite subject observations.")
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.corrcoef(values)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
        if correlation_mode == "absolute":
            correlation_similarity = np.abs(correlation)
        else:
            correlation_similarity = (correlation + 1.0) / 2.0
        sources["correlation"] = np.clip(correlation_similarity, 0.0, 1.0)

    if feature_families is not None:
        families = np.asarray(feature_families)
        if families.ndim != 1 or families.size != n_features or np.any(families == None):  # noqa: E711
            raise ValueError("feature_families must contain one non-missing label per feature.")
        sources["family"] = (families[:, None] == families[None, :]).astype(float)

    if overlap is not None:
        sources["overlap"] = _validate_matrix(overlap, n_features, "overlap")
    if custom_similarity is not None:
        sources["custom"] = _validate_matrix(
            custom_similarity, n_features, "custom_similarity"
        )

    if source_weights is None:
        weights_by_source = {name: 1.0 for name in sources}
    else:
        unknown = set(source_weights) - set(sources)
        if unknown:
            raise ValueError(f"Weights supplied for unavailable similarity sources: {sorted(unknown)}.")
        weights_by_source = {name: float(source_weights.get(name, 0.0)) for name in sources}
        if any(not np.isfinite(value) or value < 0 for value in weights_by_source.values()):
            raise ValueError("source_weights must be finite and nonnegative.")
        if sources and sum(weights_by_source.values()) <= 0:
            raise ValueError("At least one available similarity source must have positive weight.")

    structural = np.zeros((n_features, n_features), dtype=float)
    if sources:
        source_total = sum(weights_by_source.values())
        for name, matrix in sources.items():
            source_weight = weights_by_source[name]
            if source_weight == 0:
                continue
            structural += (source_weight / source_total) * matrix
    else:
        structural.fill(1.0)

    if similarity_floor:
        structural = np.maximum(structural, similarity_floor)
    neighborhood_weights, normalization_diagnostics = normalize_similarity_weights(
        structural, trust=trust, exclude_diagonal=exclude_diagonal
    )
    diagnostics: dict[str, object] = {
        "sources": tuple(name for name, value in weights_by_source.items() if value > 0),
        "source_weights": weights_by_source,
        "correlation_mode": correlation_mode,
        "bandwidth_mm": resolved_bandwidth,
        "trust": float(trust),
        "exclude_diagonal": bool(exclude_diagonal),
        **normalization_diagnostics,
    }
    return neighborhood_weights, diagnostics
