"""Shared ComBat input preparation, standardization, and correction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .global_prior import construct_global_priors
from .local_likelihood import construct_local_likelihood_priors
from .mixture import fit_mixture_priors
from .naive_prior import construct_naive_priors
from .posterior import iterative_posterior, mixture_posterior


def _prepare_covariates(covariates: object, n_subjects: int) -> np.ndarray:
    if covariates is None:
        return np.zeros((n_subjects, 0), dtype=float)
    if isinstance(covariates, pd.DataFrame):
        frame = covariates.reset_index(drop=True)
        if frame.shape[0] != n_subjects:
            raise ValueError("covariates must have one row per subject.")
        frame = pd.get_dummies(frame, drop_first=True, dtype=float)
        matrix = frame.to_numpy(dtype=float)
    else:
        matrix = np.asarray(covariates)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] != n_subjects:
            raise ValueError("covariates must have shape (subjects, covariates).")
        if not np.issubdtype(matrix.dtype, np.number):
            frame = pd.get_dummies(pd.DataFrame(matrix), drop_first=True, dtype=float)
            matrix = frame.to_numpy(dtype=float)
        else:
            matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariates must contain only finite values.")
    return matrix


def _stable_levels(batch: np.ndarray) -> list[object]:
    unique = pd.unique(batch)
    try:
        return sorted(unique.tolist())
    except TypeError:
        return sorted(unique.tolist(), key=lambda value: (type(value).__name__, str(value)))


def _fit_standardization(
    data: np.ndarray,
    batch: np.ndarray,
    covariates: object,
) -> dict[str, object]:
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional (features, subjects) array.")
    if data.shape[0] < 2 or data.shape[1] < 4:
        raise ValueError("At least two features and four subjects are required.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")
    batch = np.asarray(batch).ravel()
    if batch.size != data.shape[1] or pd.isna(batch).any():
        raise ValueError("batch must have one non-missing label per subject.")
    levels = _stable_levels(batch)
    if len(levels) < 2:
        raise ValueError("At least two batches are required.")
    batch_indices = [np.flatnonzero(batch == level) for level in levels]
    if any(index.size < 2 for index in batch_indices):
        raise ValueError("Each batch must have at least two subjects to estimate variance.")
    batch_design = np.column_stack([batch == level for level in levels]).astype(float)
    covariate_matrix = _prepare_covariates(covariates, data.shape[1])
    design = np.column_stack((batch_design, covariate_matrix))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Batch and covariate design is rank deficient or confounded.")
    coefficients = np.linalg.lstsq(design, data.T, rcond=None)[0]
    fitted = (design @ coefficients).T
    residuals = data - fitted
    residual_degrees_of_freedom = data.shape[1] - np.linalg.matrix_rank(design)
    if residual_degrees_of_freedom <= 0:
        raise ValueError("The design leaves no residual degrees of freedom.")
    pooled_variance = np.sum(residuals**2, axis=1) / residual_degrees_of_freedom
    positive = pooled_variance[pooled_variance > np.finfo(float).eps]
    if positive.size == 0:
        raise ValueError("All features have zero residual variance.")
    variance_floor = max(float(np.median(positive)) * 1e-12, np.finfo(float).tiny)
    floored = pooled_variance < variance_floor
    pooled_variance = np.maximum(pooled_variance, variance_floor)
    grand_mean = (np.array([indices.size for indices in batch_indices]) / data.shape[1]) @ coefficients[: len(levels)]
    stand_mean = np.tile(grand_mean[:, None], (1, data.shape[1]))
    if covariate_matrix.shape[1]:
        stand_mean += (covariate_matrix @ coefficients[len(levels) :]).T
    standardized = (data - stand_mean) / np.sqrt(pooled_variance)[:, None]
    gamma_hat = np.stack([np.mean(standardized[:, indices], axis=1) for indices in batch_indices])
    delta_hat = np.stack(
        [np.var(standardized[:, indices], axis=1, ddof=1) for indices in batch_indices]
    )
    return {
        "data": data,
        "batch": batch,
        "levels": np.asarray(levels),
        "batches": batch_indices,
        "design": design,
        "coefficients": coefficients,
        "standardized": standardized,
        "stand_mean": stand_mean,
        "pooled_variance": pooled_variance,
        "gamma_hat": gamma_hat,
        "delta_hat": delta_hat,
        "variance_floor_applied": floored,
    }


def fit_combat(
    data: object,
    batch: object,
    covariates: object = None,
    prior_mode: str = "global",
    similarity: np.ndarray | None = None,
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit and apply global or naive similarity-weighted parametric ComBat.

    Parameters
    ----------
    data:
        Finite numeric array in features-by-subjects orientation.
    batch:
        One batch label per subject; output batch order is stable sorted order.
    covariates:
        Optional numeric matrix or DataFrame with one row per subject.
    prior_mode:
        ``"global"`` or ``"naive"``. The latter is the original weighted-moment
        baseline, not the locally weighted likelihood method.
    similarity:
        Optional nonnegative feature-by-feature matrix used by naive mode.
    options:
        Posterior controls and naive similarity controls.
    """
    options = {} if options is None else dict(options)
    if prior_mode not in {"global", "naive", "local_likelihood", "mixture"}:
        raise ValueError("prior_mode must be 'global', 'naive', 'local_likelihood', or 'mixture'.")
    data_array = np.asarray(data, dtype=float)
    state = _fit_standardization(data_array, np.asarray(batch), covariates)
    gamma_hat = state["gamma_hat"]
    delta_hat = state["delta_hat"]
    global_values = construct_global_priors(gamma_hat, delta_hat)
    gamma_global, variance_global, shape_global, scale_global, global_diagnostics = global_values

    mixture_posterior_diagnostics = None
    if prior_mode == "naive":
        prior_details = construct_naive_priors(
            state["standardized"],
            gamma_hat,
            delta_hat,
            state["batches"],
            (gamma_global, variance_global, shape_global, scale_global),
            methods=options.get("weight_methods"),
            options=options.get("weight_options"),
            similarity=similarity,
            legacy_compat=bool(options.get("legacy_compat", True)),
        )
        gamma_mean = prior_details["gamma_mean"]
        gamma_variance = prior_details["gamma_variance"]
        delta_shape = prior_details["delta_shape"]
        delta_scale = prior_details["delta_scale"]
    elif prior_mode == "local_likelihood":
        prior_details = construct_local_likelihood_priors(
            state["standardized"],
            gamma_hat,
            delta_hat,
            state["batches"],
            (gamma_global, variance_global, shape_global, scale_global),
            similarity=similarity,
            options=options.get("local_likelihood_options"),
        )
        gamma_mean = prior_details["gamma_mean"]
        gamma_variance = prior_details["gamma_variance"]
        delta_shape = prior_details["delta_shape"]
        delta_scale = prior_details["delta_scale"]
    elif prior_mode == "mixture":
        mixture_options = options.get("mixture_options", {})
        prior_details = fit_mixture_priors(
            gamma_hat,
            delta_hat,
            np.array([len(indices) for indices in state["batches"]]),
            n_components=int(mixture_options.get("n_components", 2)),
            memberships=mixture_options.get("memberships"),
            feature_families=mixture_options.get("feature_families"),
            overlap=mixture_options.get("overlap"),
            similarity=similarity,
            options=mixture_options,
        )
        mixture_memberships = prior_details["memberships"]
        gamma_mean = prior_details["gamma_mean"]
        gamma_variance = prior_details["gamma_variance"]
        delta_shape = prior_details["delta_shape"]
        delta_scale = prior_details["delta_scale"]
    else:
        prior_details = {
            "gamma_mean": np.broadcast_to(gamma_global[:, None], gamma_hat.shape).copy(),
            "gamma_variance": np.broadcast_to(variance_global[:, None], gamma_hat.shape).copy(),
            "delta_shape": np.broadcast_to(shape_global[:, None], gamma_hat.shape).copy(),
            "delta_scale": np.broadcast_to(scale_global[:, None], gamma_hat.shape).copy(),
        }
        gamma_mean = prior_details["gamma_mean"]
        gamma_variance = prior_details["gamma_variance"]
        delta_shape = prior_details["delta_shape"]
        delta_scale = prior_details["delta_scale"]

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    posterior_diagnostics = {}
    if prior_mode == "mixture":
        for batch_index, sample_indices in enumerate(state["batches"]):
            component_means = np.broadcast_to(
                gamma_mean[batch_index][None, :],
                (gamma_hat.shape[1], gamma_mean.shape[1]),
            )
            component_variances = np.broadcast_to(
                gamma_variance[batch_index][None, :], component_means.shape
            )
            component_shapes = np.broadcast_to(
                delta_shape[batch_index][None, :], component_means.shape
            )
            component_scales = np.broadcast_to(
                delta_scale[batch_index][None, :], component_means.shape
            )
            mixture_result = mixture_posterior(
                state["standardized"][:, sample_indices],
                mixture_memberships,
                component_means,
                component_variances,
                component_shapes,
                component_scales,
                quadrature_bounds=tuple(
                    options.get("mixture_quadrature_bounds", (-30.0, 30.0))
                ),
            )
            gamma_star[batch_index] = mixture_result["posterior_gamma_mean"]
            delta_star[batch_index] = mixture_result["posterior_delta_mean"]
            posterior_diagnostics[str(state["levels"][batch_index])] = mixture_result[
                "diagnostics"
            ]
            if mixture_posterior_diagnostics is None:
                mixture_posterior_diagnostics = {}
            mixture_posterior_diagnostics[str(state["levels"][batch_index])] = mixture_result
    else:
        for batch_index, sample_indices in enumerate(state["batches"]):
            gamma_star[batch_index], delta_star[batch_index], posterior_diagnostics[str(state["levels"][batch_index])] = iterative_posterior(
                state["standardized"][:, sample_indices],
                gamma_hat[batch_index],
                delta_hat[batch_index],
                gamma_mean[batch_index],
                gamma_variance[batch_index],
                delta_shape[batch_index],
                delta_scale[batch_index],
                tolerance=float(options.get("tolerance", 1e-6)),
                max_iterations=int(options.get("max_iterations", 200)),
            )

    corrected = state["standardized"].copy()
    gamma_correction = bool(options.get("gamma_correction", True))
    delta_correction = bool(options.get("delta_correction", True))
    for batch_index, sample_indices in enumerate(state["batches"]):
        if gamma_correction:
            corrected[:, sample_indices] -= gamma_star[batch_index, :, None]
        if delta_correction:
            corrected[:, sample_indices] /= np.sqrt(delta_star[batch_index])[:, None]
    harmonized = corrected * np.sqrt(state["pooled_variance"])[:, None] + state["stand_mean"]
    if not np.all(np.isfinite(harmonized)):
        raise FloatingPointError("ComBat correction produced non-finite values.")

    return {
        "harmonized": harmonized,
        "gamma_hat": gamma_hat,
        "delta_hat": delta_hat,
        "gamma_star": gamma_star,
        "delta_star": delta_star,
        "prior_details": prior_details,
        "diagnostics": {
            "prior_mode": prior_mode,
            "batch_levels": state["levels"],
            "batch_sizes": np.array([len(indices) for indices in state["batches"]]),
            "pooled_variance": state["pooled_variance"],
            "variance_floor_applied": state["variance_floor_applied"],
            "global_prior": global_diagnostics,
            "posterior": posterior_diagnostics,
            "mixture_posterior": mixture_posterior_diagnostics,
        },
    }
