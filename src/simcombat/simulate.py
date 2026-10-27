"""Seeded semi-realistic MRI-inspired feature and batch-effect simulations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_FEATURE_FAMILIES = ("cortical_thickness", "cortical_volume", "subcortical_volume", "intensity")


def simulate_dataset(
    *,
    seed: int = 0,
    n_features: int = 48,
    batch_sizes: tuple[int, ...] = (32, 32, 32, 32),
    n_components: int = 2,
    effect_heterogeneity: str = "heterogeneous",
    similarity_informativeness: str = "informative",
    feature_correlation: float = 0.35,
    batch_effect_strength: float = 0.35,
    biological_batch_association: float = 0.25,
    zero_batch_effect: bool = False,
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Generate MRI-like feature measurements and exact paired counterfactual truth.

    The sample model is ``Y0 = biological_mean + feature_scale * epsilon`` and
    ``Y = biological_mean + feature_scale * (gamma + sqrt(delta) * epsilon)``.
    Consequently, Y0 and Y share the same biological signal and Gaussian noise;
    the batch effect is added on the native measurement scale through the saved
    feature scales. ``gamma_true`` and ``delta_true`` are standardized effects.
    """
    options = {} if options is None else dict(options)
    if n_features < max(8, n_components * 2):
        raise ValueError("n_features must be at least 8 and at least twice n_components.")
    if len(batch_sizes) < 3 or any(size < 4 for size in batch_sizes):
        raise ValueError("At least three batches with four or more subjects each are required.")
    if n_components < 1 or n_components > 4:
        raise ValueError("n_components must lie in [1, 4].")
    if effect_heterogeneity not in {"homogeneous", "heterogeneous"}:
        raise ValueError("effect_heterogeneity must be 'homogeneous' or 'heterogeneous'.")
    if similarity_informativeness not in {"informative", "partial", "uninformative", "misleading"}:
        raise ValueError("Unknown similarity_informativeness setting.")
    if not 0.0 <= feature_correlation < 0.95:
        raise ValueError("feature_correlation must lie in [0, 0.95).")
    if batch_effect_strength < 0 or not 0 <= biological_batch_association < 0.9:
        raise ValueError("Invalid batch-effect strength or biological association.")

    rng = np.random.default_rng(seed)
    n_batches = len(batch_sizes)
    n_subjects = int(sum(batch_sizes))
    batch = np.repeat(np.arange(n_batches), batch_sizes)
    batch_frame = pd.DataFrame({"batch": batch})

    region_index = np.arange(n_features)
    family_count = len(_FEATURE_FAMILIES)
    family_index = (region_index * family_count // n_features).astype(int)
    feature_families = np.asarray(_FEATURE_FAMILIES)[family_index]
    hemisphere = np.where(region_index % 2 == 0, "left", "right")
    region_position = region_index % int(np.ceil(n_features / family_count))
    coordinates = np.column_stack(
        (
            18.0 * family_index + 2.5 * np.cos(region_position / 3.0),
            3.0 * region_position,
            7.0 * np.sin(region_position / 4.0) + 2.0 * (hemisphere == "right"),
        )
    )
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    positive_distances = distances[distances > 0]
    spatial_bandwidth = float(np.median(positive_distances)) if positive_distances.size else 1.0
    spatial_similarity = np.exp(-(distances**2) / (2.0 * spatial_bandwidth**2))
    overlap = (feature_families[:, None] == feature_families[None, :]).astype(float)
    np.fill_diagonal(overlap, 1.0)

    # Ensure every batch retains subjects across the continuous covariate range.
    age = rng.uniform(18.0, 85.0, n_subjects)
    binary_covariate = rng.binomial(1, 0.48, n_subjects).astype(float)
    group = rng.binomial(1, 0.4, n_subjects).astype(float)
    age = age + biological_batch_association * (batch - np.mean(batch)) * 7.0
    covariates = pd.DataFrame(
        {
            "age": (age - age.mean()) / age.std(ddof=0),
            "binary": binary_covariate,
            "group": group,
        }
    )

    latent_count = max(3, min(12, n_features // 4))
    latent_scores = rng.normal(size=(latent_count, n_subjects))
    loadings = rng.normal(size=(n_features, latent_count))
    group_count = max(1, n_components)
    for feature_index in range(n_features):
        loadings[feature_index, feature_index % latent_count] += 1.5
        if feature_index % 7 == 0:
            loadings[feature_index, (feature_index + 1) % latent_count] -= 0.8
    correlated_signal = loadings @ latent_scores / np.sqrt(latent_count)
    feature_noise = rng.normal(size=(n_features, n_subjects))
    biological_noise = (
        np.sqrt(feature_correlation) * correlated_signal
        + np.sqrt(1.0 - feature_correlation) * feature_noise
    )

    family_mean = np.array([0.0, 12.0, 850.0, 65.0])
    family_scale = np.array([0.12, 95.0, 70.0, 18.0])
    feature_scale = family_scale[family_index] * rng.lognormal(mean=0.0, sigma=0.18, size=n_features)
    biological_beta = rng.normal(0.0, 0.25, size=(3, n_features))
    biological_mean = family_mean[family_index, None] + feature_scale[:, None] * (
        biological_beta[0, :, None] * covariates["age"].to_numpy()[None, :]
        + biological_beta[1, :, None] * covariates["binary"].to_numpy()[None, :]
        + biological_beta[2, :, None] * covariates["group"].to_numpy()[None, :]
    )
    epsilon = biological_noise
    y0 = biological_mean + feature_scale[:, None] * epsilon

    if n_components == 1:
        memberships_true = np.ones((n_features, 1))
    else:
        group_assignment = np.minimum(
            (region_index * n_components // n_features).astype(int), n_components - 1
        )
        memberships_true = np.full((n_features, n_components), 0.04 / (n_components - 1))
        memberships_true[np.arange(n_features), group_assignment] = 0.96
    if similarity_informativeness == "misleading" and n_components > 1:
        assignments = np.argmax(memberships_true, axis=1)
        assignments = (assignments + 1) % n_components
        memberships_true = np.full((n_features, n_components), 0.04 / (n_components - 1))
        memberships_true[np.arange(n_features), assignments] = 0.96

    gamma_true = np.zeros((n_batches, n_features))
    delta_true = np.ones((n_batches, n_features))
    if not zero_batch_effect:
        for batch_index in range(n_batches):
            if effect_heterogeneity == "homogeneous":
                batch_shift = (batch_index - (n_batches - 1) / 2.0) * batch_effect_strength / max(n_batches - 1, 1)
                gamma_true[batch_index] = batch_shift
                delta_true[batch_index] = 1.0 + 0.15 * batch_shift
            else:
                component_gamma = np.linspace(-1.0, 1.0, n_components) * batch_effect_strength
                component_scale = np.exp(np.linspace(-0.35, 0.35, n_components) * batch_effect_strength)
                for feature_index in range(n_features):
                    membership = memberships_true[feature_index]
                    group_shift = float(membership @ component_gamma)
                    group_scale = float(membership @ component_scale)
                    gamma_true[batch_index, feature_index] = (
                        group_shift * (1.0 + 0.08 * batch_index)
                        + rng.normal(0.0, 0.1 * batch_effect_strength)
                    )
                    delta_true[batch_index, feature_index] = max(
                        0.2,
                        group_scale * np.exp(rng.normal(0.0, 0.08)),
                    )

    batch_proportions = np.asarray(batch_sizes, dtype=float) / n_subjects
    pooled_gamma = batch_proportions @ gamma_true
    pooled_delta = batch_proportions @ delta_true
    gamma_standardized = (gamma_true - pooled_gamma[None, :]) / np.sqrt(pooled_delta)[None, :]
    delta_standardized = delta_true / pooled_delta[None, :]
    gamma_native_reference = (gamma_true - pooled_gamma[None, :]) * feature_scale[None, :]

    observed = np.empty_like(y0)
    for batch_index in range(n_batches):
        subject_indices = np.flatnonzero(batch == batch_index)
        observed[:, subject_indices] = biological_mean[:, subject_indices] + feature_scale[:, None] * (
            gamma_true[batch_index, :, None]
            + np.sqrt(delta_true[batch_index, :, None]) * epsilon[:, subject_indices]
        )

    if similarity_informativeness == "uninformative":
        structural_similarity = np.ones((n_features, n_features))
    elif similarity_informativeness == "partial":
        noise = rng.uniform(0.0, 1.0, size=(n_features, n_features))
        noise = 0.5 * (noise + noise.T)
        structural_similarity = 0.25 * spatial_similarity + 0.25 * overlap + 0.5 * noise
    elif similarity_informativeness == "misleading":
        structural_similarity = 0.5 * spatial_similarity + 0.5 * (1.0 - overlap)
    else:
        structural_similarity = 0.5 * spatial_similarity + 0.5 * overlap
    np.fill_diagonal(structural_similarity, 1.0)

    feature_metadata = pd.DataFrame(
        {
            "feature_id": [f"parcel_{index:03d}" for index in region_index],
            "family": feature_families,
            "hemisphere": hemisphere,
            "region_position": region_position,
        }
    )
    return {
        "observed": observed,
        "batch_free": y0,
        "batch": batch,
        "covariates": covariates,
        "design": np.column_stack((np.ones(n_subjects), covariates.to_numpy())),
        "true": {
            "gamma": gamma_true,
            "delta": delta_true,
            "memberships": memberships_true,
            "biological_mean": biological_mean,
            "epsilon": epsilon,
            "feature_scale": feature_scale,
            "biological_coefficients": biological_beta,
            "standardized_gamma": gamma_standardized,
            "standardized_delta": delta_standardized,
            "pooled_reference_gamma": pooled_gamma,
            "pooled_reference_delta": pooled_delta,
            "native_gamma_shift": gamma_native_reference,
        },
        "features": {
            "metadata": feature_metadata,
            "families": feature_families,
            "coordinates_mm": coordinates,
            "overlap": overlap,
            "spatial_similarity": spatial_similarity,
            "structural_similarity": structural_similarity,
        },
        "batch_sizes": np.asarray(batch_sizes),
        "seed": int(seed),
        "scenario": {
            "effect_heterogeneity": effect_heterogeneity,
            "similarity_informativeness": similarity_informativeness,
            "feature_correlation": float(feature_correlation),
            "n_components": int(n_components),
            "zero_batch_effect": bool(zero_batch_effect),
            "paired_truth_uses_same_noise": True,
            "reference_convention": "weighted_pooled_batch_mean",
        },
    }
