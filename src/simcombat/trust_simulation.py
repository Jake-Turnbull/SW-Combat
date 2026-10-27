"""Factorial simulations for trust-calibrated local-prior ComBat."""

from __future__ import annotations

import numpy as np

SHAPE_FAMILIES = ("unimodal", "bimodal", "trimodal", "zero_centered_bimodal")


def _draw_effect_shape(rng: np.random.Generator, shape: str, size: int) -> np.ndarray:
    if shape not in SHAPE_FAMILIES:
        raise ValueError(f"shape must be one of {SHAPE_FAMILIES}.")
    if shape == "unimodal":
        values = rng.normal(0.0, 0.65, size)
    elif shape == "bimodal":
        component = rng.choice(2, size=size, p=(0.65, 0.35))
        centers = np.array([-0.8, 1.0])
        values = centers[component] + rng.normal(0.0, 0.18, size)
    elif shape == "trimodal":
        component = rng.choice(3, size=size, p=(0.25, 0.5, 0.25))
        centers = np.array([-1.0, 0.0, 1.0])
        values = centers[component] + rng.normal(0.0, 0.14, size)
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=size)
        values = signs * 0.85 + rng.normal(0.0, 0.14, size)
    values -= np.mean(values)
    scale = np.std(values)
    return values / scale if scale > 0 else values


def simulate_trust_scenario(
    *,
    seed: int,
    mean_shape: str,
    log_delta_shape: str,
    n_features: int = 32,
    batch_sizes: tuple[int, ...] = (16, 16, 16),
    batch_effect_strength: float = 0.45,
    sample_effect_sd: float = 0.15,
    covariate_noise_sd: float = 0.15,
    feature_correlation: float = 0.3,
) -> dict[str, object]:
    """Generate one crossed additive/log-scale shape scenario with realized truth.

    Batch and sample effects use ``delta`` as a variance multiplier, so residuals
    are scaled by ``sqrt(delta)``. Sample-specific batch effects and covariate
    coefficient noise are returned as truth arrays rather than discarded.
    """
    if n_features < 4 or len(batch_sizes) < 2 or any(size < 3 for size in batch_sizes):
        raise ValueError("Require at least four features and two batches of size three or more.")
    if batch_effect_strength < 0 or sample_effect_sd < 0 or covariate_noise_sd < 0:
        raise ValueError("Effect strengths and noise scales must be nonnegative.")
    if not 0.0 <= feature_correlation < 1.0:
        raise ValueError("feature_correlation must lie in [0, 1).")
    if mean_shape not in SHAPE_FAMILIES or log_delta_shape not in SHAPE_FAMILIES:
        raise ValueError(f"mean_shape and log_delta_shape must be in {SHAPE_FAMILIES}.")

    rng = np.random.default_rng(seed)
    n_batches = len(batch_sizes)
    n_subjects = int(sum(batch_sizes))
    batch = np.repeat(np.arange(n_batches), batch_sizes)
    batch_indices = [np.flatnonzero(batch == index) for index in range(n_batches)]
    batch_contrast = np.linspace(-1.0, 1.0, n_batches)
    feature_scale = rng.lognormal(mean=0.0, sigma=0.35, size=n_features)

    age = rng.normal(size=n_subjects)
    age -= np.mean(age)
    age /= max(float(np.std(age)), np.finfo(float).eps)
    covariate_beta = rng.normal(0.0, 0.35, size=n_features)
    covariate_effect_noise = rng.normal(
        0.0, covariate_noise_sd, size=(n_features, n_subjects)
    )
    realized_covariate_effect = (
        covariate_beta[:, None] + covariate_effect_noise
    ) * age[None, :]

    latent = rng.normal(size=(max(2, n_features // 6), n_subjects))
    loadings = rng.normal(size=(n_features, latent.shape[0]))
    correlated_residual = loadings @ latent / np.sqrt(latent.shape[0])
    independent_residual = rng.normal(size=(n_features, n_subjects))
    residual = (
        np.sqrt(feature_correlation) * correlated_residual
        + np.sqrt(1.0 - feature_correlation) * independent_residual
    )
    residual /= np.std(residual, axis=1, keepdims=True)

    mean_feature_effect = _draw_effect_shape(rng, mean_shape, n_features)
    log_delta_feature_effect = _draw_effect_shape(rng, log_delta_shape, n_features)
    gamma_batch = batch_effect_strength * batch_contrast[:, None] * mean_feature_effect[None, :]
    log_delta_batch = (
        batch_effect_strength * batch_contrast[:, None] * log_delta_feature_effect[None, :]
    )
    gamma_sample = np.zeros((n_features, n_subjects), dtype=float)
    log_delta_sample = np.zeros_like(gamma_sample)
    for batch_index, indices in enumerate(batch_indices):
        gamma_sample[:, indices] = gamma_batch[batch_index, :, None] + rng.normal(
            0.0, sample_effect_sd, size=(n_features, indices.size)
        )
        log_delta_sample[:, indices] = log_delta_batch[batch_index, :, None] + rng.normal(
            0.0, sample_effect_sd, size=(n_features, indices.size)
        )
    delta_sample = np.exp(log_delta_sample)

    biological_mean = feature_scale[:, None] * realized_covariate_effect
    batch_free = biological_mean + feature_scale[:, None] * residual
    observed = biological_mean + feature_scale[:, None] * (
        gamma_sample + np.sqrt(delta_sample) * residual
    )
    residual_similarity = np.corrcoef(residual)
    structural_similarity = np.nan_to_num(np.abs(residual_similarity), nan=0.0)
    structural_similarity = np.maximum(structural_similarity, 0.01)
    np.fill_diagonal(structural_similarity, 1.0)

    return {
        "observed": observed,
        "batch_free": batch_free,
        "batch": batch,
        "covariates": age[:, None],
        "batches": batch_indices,
        "features": {
            "feature_scale": feature_scale,
            "structural_similarity": structural_similarity,
            "residual_correlation": residual_similarity,
        },
        "true": {
            "mean_shape": mean_shape,
            "log_delta_shape": log_delta_shape,
            "mean_feature_effect": mean_feature_effect,
            "log_delta_feature_effect": log_delta_feature_effect,
            "gamma_batch": gamma_batch,
            "log_delta_batch": log_delta_batch,
            "gamma_sample": gamma_sample,
            "log_delta_sample": log_delta_sample,
            "delta_sample": delta_sample,
            "covariate_beta": covariate_beta,
            "covariate_effect_noise": covariate_effect_noise,
            "realized_covariate_effect": realized_covariate_effect,
            "residual": residual,
        },
        "batch_sizes": np.asarray(batch_sizes, dtype=int),
        "seed": int(seed),
    }
