"""Global parametric ComBat prior estimation."""

from __future__ import annotations

import numpy as np


def _inverse_gamma_moments(values: np.ndarray) -> tuple[float, float, bool]:
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    if mean <= 0 or not np.isfinite(mean):
        raise ValueError("Delta estimates must have a positive finite mean.")
    if variance <= np.finfo(float).eps * max(1.0, mean * mean):
        shape = 1e6
        scale = mean * (shape - 1.0)
        return shape, scale, True
    shape = (2.0 * variance + mean * mean) / variance
    scale = (mean * variance + mean**3) / variance
    return float(shape), float(scale), False


def construct_global_priors(
    gamma_hat: np.ndarray, delta_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Estimate batch-specific Normal and inverse-gamma hyperparameters."""
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    delta_hat = np.asarray(delta_hat, dtype=float)
    if gamma_hat.ndim != 2 or delta_hat.shape != gamma_hat.shape:
        raise ValueError("gamma_hat and delta_hat must share shape (batches, features).")
    if gamma_hat.shape[1] < 2:
        raise ValueError("At least two features are required to estimate global priors.")

    n_batches = gamma_hat.shape[0]
    gamma_mean = np.mean(gamma_hat, axis=1)
    gamma_variance = np.var(gamma_hat, axis=1, ddof=1)
    gamma_floored = gamma_variance <= np.finfo(float).eps
    gamma_variance = np.maximum(gamma_variance, 1e-6)
    delta_shape = np.empty(n_batches)
    delta_scale = np.empty(n_batches)
    delta_fallback = np.zeros(n_batches, dtype=bool)
    for batch_index in range(n_batches):
        delta_shape[batch_index], delta_scale[batch_index], delta_fallback[batch_index] = (
            _inverse_gamma_moments(delta_hat[batch_index])
        )
    diagnostics = {
        "gamma_variance_floored": gamma_floored,
        "delta_moment_fallback": delta_fallback,
    }
    return gamma_mean, gamma_variance, delta_shape, delta_scale, diagnostics
