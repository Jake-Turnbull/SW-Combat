"""Primary harmonization metrics and replicate-level paired summaries."""

from __future__ import annotations

import numpy as np


def primary_metrics(
    harmonized: np.ndarray,
    batch_free: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    gamma_true: np.ndarray,
    delta_true: np.ndarray,
    feature_scale: np.ndarray,
) -> dict[str, float]:
    """Compute primary paired truth metrics in native and standardized units."""
    harmonized = np.asarray(harmonized, dtype=float)
    batch_free = np.asarray(batch_free, dtype=float)
    feature_scale = np.asarray(feature_scale, dtype=float)
    if harmonized.shape != batch_free.shape or harmonized.ndim != 2:
        raise ValueError("harmonized and batch_free must share (features, subjects) shape.")
    if feature_scale.shape != (harmonized.shape[0],) or np.any(feature_scale <= 0):
        raise ValueError("feature_scale must be positive with one value per feature.")
    if not all(np.all(np.isfinite(value)) for value in (harmonized, batch_free, feature_scale)):
        raise ValueError("Metric inputs must be finite.")

    error = harmonized - batch_free
    standardized_error = error / feature_scale[:, None]
    variance_harmonized = np.var(harmonized, axis=1, ddof=1)
    variance_truth = np.var(batch_free, axis=1, ddof=1)
    valid = variance_truth > np.finfo(float).tiny
    variance_ratio = np.ones_like(variance_truth)
    variance_ratio[valid] = variance_harmonized[valid] / variance_truth[valid]
    epsilon = np.finfo(float).tiny
    gamma_true = np.asarray(gamma_true, dtype=float)
    delta_true = np.asarray(delta_true, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    delta_hat = np.asarray(delta_hat, dtype=float)
    if gamma_hat.shape != gamma_true.shape or delta_hat.shape != delta_true.shape:
        raise ValueError("Estimated and true batch-effect arrays must have matching shapes.")

    return {
        "rmse_native": float(np.sqrt(np.mean(error**2))),
        "mae_native": float(np.mean(np.abs(error))),
        "rmse_standardized": float(np.sqrt(np.mean(standardized_error**2))),
        "mae_standardized": float(np.mean(np.abs(standardized_error))),
        "mean_signed_bias_native": float(np.mean(error)),
        "mean_absolute_bias_native": float(np.mean(np.abs(np.mean(error, axis=1)))),
        "mean_signed_bias_standardized": float(np.mean(standardized_error)),
        "mean_absolute_bias_standardized": float(np.mean(np.abs(np.mean(standardized_error, axis=1)))),
        "variance_ratio_mean": float(np.mean(variance_ratio)),
        "absolute_log_variance_ratio": float(np.mean(np.abs(np.log(np.maximum(variance_ratio, epsilon))))),
        "variance_mae": float(np.mean(np.abs(variance_harmonized - variance_truth))),
        "gamma_rmse": float(np.sqrt(np.mean((gamma_hat - gamma_true) ** 2))),
        "gamma_mae": float(np.mean(np.abs(gamma_hat - gamma_true))),
        "delta_log_rmse": float(
            np.sqrt(np.mean((np.log(np.maximum(delta_hat, epsilon)) - np.log(np.maximum(delta_true, epsilon))) ** 2))
        ),
        "delta_log_mae": float(
            np.mean(np.abs(np.log(np.maximum(delta_hat, epsilon)) - np.log(np.maximum(delta_true, epsilon))))
        ),
    }


def paired_bootstrap_interval(
    replicate_differences: np.ndarray,
    *,
    seed: int = 0,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile interval resampling replicates, not features."""
    values = np.asarray(replicate_differences, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("replicate_differences must be a nonempty finite vector.")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence lie in (0, 1).")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = np.mean(values[sampled], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(np.mean(values)), float(lower), float(upper)
