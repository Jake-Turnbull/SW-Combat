"""Uncertainty-aware locally weighted likelihood prior estimation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from .similarity import build_similarity, normalize_similarity_weights


def normal_mean_variance_log_likelihood(
    means: np.ndarray,
    observation_variances: np.ndarray,
    weights: np.ndarray,
    prior_mean: float,
    prior_variance: float,
) -> tuple[float, float]:
    """Weighted Gaussian marginal log likelihood and profiled prior mean."""
    means = np.asarray(means, dtype=float)
    observation_variances = np.asarray(observation_variances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if means.ndim != 1 or observation_variances.shape != means.shape or weights.shape != means.shape:
        raise ValueError("means, observation_variances, and weights must share one-dimensional shape.")
    if not all(np.all(np.isfinite(value)) for value in (means, observation_variances, weights)):
        raise ValueError("Normal marginal likelihood inputs must be finite.")
    if np.any(observation_variances < 0) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Observation variances and weights must be nonnegative with positive total weight.")
    if prior_variance <= 0 or not np.isfinite(prior_variance):
        raise ValueError("prior_variance must be positive and finite.")
    total_variance = prior_variance + observation_variances
    precision = weights / total_variance
    profiled_mean = float(np.sum(precision * means) / np.sum(precision))
    log_likelihood = -0.5 * float(
        np.sum(weights * (np.log(total_variance) + (means - profiled_mean) ** 2 / total_variance))
    )
    return log_likelihood, profiled_mean


def inverse_gamma_variance_log_likelihood(
    sample_variances: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: np.ndarray,
    shape: float,
    scale: float,
) -> np.ndarray:
    """Integrated log density of Gaussian sample variances under an IG prior.

    The inverse-gamma convention is shape/scale. This is the exact marginal
    density of the unbiased sample variance under the Gaussian chi-square
    sampling model stated in the implementation plan.
    """
    sample_variances = np.asarray(sample_variances, dtype=float)
    degrees_of_freedom = np.asarray(degrees_of_freedom, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if sample_variances.ndim != 1 or degrees_of_freedom.shape != sample_variances.shape or weights.shape != sample_variances.shape:
        raise ValueError("Variance likelihood inputs must share one-dimensional shape.")
    if not all(np.all(np.isfinite(value)) for value in (sample_variances, degrees_of_freedom, weights)):
        raise ValueError("Variance likelihood inputs must be finite.")
    if np.any(sample_variances <= 0) or np.any(degrees_of_freedom < 1) or np.any(weights < 0):
        raise ValueError("Sample variances and degrees of freedom must be positive; weights nonnegative.")
    if shape <= 2 or scale <= 0 or not np.isfinite(shape + scale):
        raise ValueError("Finite-moment inverse-gamma priors require shape > 2 and scale > 0.")
    half_df = degrees_of_freedom / 2.0
    return (
        gammaln(shape + half_df)
        - gammaln(shape)
        - gammaln(half_df)
        + half_df * np.log(half_df)
        + shape * np.log(scale)
        + (half_df - 1.0) * np.log(sample_variances)
        - (shape + half_df) * np.log(scale + half_df * sample_variances)
    )


def _fit_normal_prior(
    means: np.ndarray,
    observation_variances: np.ndarray,
    weights: np.ndarray,
    global_mean: float,
    global_variance: float,
    penalty: float,
) -> tuple[float, float, dict[str, object]]:
    positive_variance = observation_variances[weights > 0]
    spread = float(np.var(means[weights > 0])) if np.any(weights > 0) else global_variance
    tau_floor = max(np.finfo(float).eps, float(np.min(positive_variance)) * 1e-6) if positive_variance.size else 1e-8
    tau_ceiling = max(global_variance, spread, float(np.max(observation_variances)), tau_floor) * 1e4
    lower, upper = np.log(tau_floor), np.log(tau_ceiling)

    def objective(log_variance: float) -> float:
        variance = float(np.exp(log_variance))
        likelihood, _ = normal_mean_variance_log_likelihood(
            means, observation_variances, weights, global_mean, variance
        )
        regularizer = 0.5 * penalty * (
            log_variance - np.log(max(global_variance, tau_floor))
        ) ** 2
        return -likelihood + regularizer

    starts = sorted(set(np.clip([np.log(max(global_variance, tau_floor)), np.log(max(spread, tau_floor))], lower, upper)))
    candidates = [
        minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-8}),
        *[
            minimize(
                lambda parameter: objective(float(parameter[0])),
                np.array([start]),
                method="L-BFGS-B",
                bounds=[(lower, upper)],
            )
            for start in starts
        ],
    ]
    successful = [candidate for candidate in candidates if candidate.success and np.isfinite(candidate.fun)]
    if not successful:
        return global_mean, global_variance, {"converged": False, "fallback": "global", "objective": np.nan}
    best = min(successful, key=lambda candidate: float(candidate.fun))
    variance = float(np.exp(float(np.ravel(best.x)[0])))
    _, mean = normal_mean_variance_log_likelihood(
        means, observation_variances, weights, global_mean, variance
    )
    step = max(abs(float(np.ravel(best.x)[0])) * 1e-4, 1e-4)
    center = float(np.ravel(best.x)[0])
    curvature = (objective(center + step) - 2.0 * objective(center) + objective(center - step)) / step**2
    return mean, variance, {
        "converged": True,
        "fallback": None,
        "objective": float(best.fun),
        "curvature": float(curvature),
        "optimizer_starts": len(starts),
    }


def _fit_inverse_gamma_prior(
    sample_variances: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: np.ndarray,
    global_shape: float,
    global_scale: float,
    penalty: float,
) -> tuple[float, float, dict[str, object]]:
    active = weights > 0
    if np.sum(active) < 2:
        return global_shape, global_scale, {"converged": False, "fallback": "global", "objective": np.nan}
    log_shape_lower, log_shape_upper = np.log(1e-4), np.log(1e5)
    log_scale_lower, log_scale_upper = np.log(1e-12), np.log(max(1.0, float(np.max(sample_variances[active]))) * 1e8)
    global_transformed = np.array(
        [np.log(max(global_shape - 2.0, 1e-4)), np.log(max(global_scale, 1e-12))]
    )
    bounds = [(log_shape_lower, log_shape_upper), (log_scale_lower, log_scale_upper)]

    def objective(parameters: np.ndarray) -> float:
        shape = 2.0 + float(np.exp(parameters[0]))
        scale = float(np.exp(parameters[1]))
        log_density = inverse_gamma_variance_log_likelihood(
            sample_variances, degrees_of_freedom, weights, shape, scale
        )
        regularizer = 0.5 * penalty * float(np.sum((parameters - global_transformed) ** 2))
        return -float(np.sum(weights * log_density)) + regularizer

    pooled_mean = float(np.average(sample_variances[active], weights=weights[active]))
    starts = [
        global_transformed,
        np.array([np.log(max(global_shape - 2.0, 1e-4)), np.log(max(pooled_mean * (global_shape - 1.0), 1e-12))]),
        np.array([np.log(2.0), np.log(max(pooled_mean * 3.0, 1e-12))]),
    ]
    candidates = [
        minimize(objective, np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds]), method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    successful = [candidate for candidate in candidates if candidate.success and np.isfinite(candidate.fun)]
    if not successful:
        return global_shape, global_scale, {"converged": False, "fallback": "global", "objective": np.nan}
    best = min(successful, key=lambda candidate: float(candidate.fun))
    shape = 2.0 + float(np.exp(best.x[0]))
    scale = float(np.exp(best.x[1]))
    return shape, scale, {
        "converged": True,
        "fallback": None,
        "objective": float(best.fun),
        "optimizer_starts": len(starts),
        "optimizer_message": str(best.message),
    }


def construct_local_likelihood_priors(
    standardized: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batches: list[np.ndarray],
    global_priors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    similarity: np.ndarray | None = None,
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit local Normal/IG priors using leave-target-out weighted marginal likelihoods.

    The Normal working likelihood treats feature mean estimation errors as
    independent with variance ``delta_hat / n_batch``. The scale likelihood
    integrates an inverse-gamma prior against the Gaussian sample-variance
    likelihood. A quadratic penalty shrinks transformed local parameters toward
    global estimates; its coefficient is explicit as ``global_prior_penalty``.
    """
    options = {} if options is None else dict(options)
    gamma_global, variance_global, shape_global, scale_global = global_priors
    n_batches, n_features = gamma_hat.shape
    penalty = float(options.get("global_prior_penalty", 0.1))
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("global_prior_penalty must be finite and nonnegative.")
    source_weights = options.get("source_weights")
    weights_all = np.zeros((n_batches, n_features, n_features), dtype=float)
    weight_diagnostics = []
    for batch_index, sample_indices in enumerate(batches):
        weights, diagnostics = build_similarity(
            n_features,
            feature_coords=options.get("feature_coords"),
            residuals=standardized[:, sample_indices] if options.get("use_residual_correlation", False) else None,
            feature_families=options.get("feature_families"),
            overlap=options.get("overlap"),
            custom_similarity=similarity,
            source_weights=source_weights,
            correlation_mode=str(options.get("correlation_mode", "absolute")),
            bandwidth_mm=options.get("bandwidth_mm"),
            trust=float(options.get("trust", 1.0)),
            exclude_diagonal=True,
            similarity_floor=float(options.get("similarity_floor", 0.0)),
        )
        weights_all[batch_index] = weights
        weight_diagnostics.append(diagnostics)

    local_gamma_mean = np.zeros((n_batches, n_features))
    local_gamma_variance = np.zeros_like(local_gamma_mean)
    local_delta_shape = np.zeros_like(local_gamma_mean)
    local_delta_scale = np.zeros_like(local_gamma_mean)
    fitting_weights = np.zeros_like(weights_all)
    diagnostics_by_target: list[list[dict[str, object]]] = []
    for batch_index, sample_indices in enumerate(batches):
        n_subjects = len(sample_indices)
        degrees = np.full(n_features, n_subjects - 1.0)
        batch_diagnostics = []
        for feature_index in range(n_features):
            weights = weights_all[batch_index, feature_index].copy()
            weights[feature_index] = 0.0
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                local_gamma_mean[batch_index, feature_index] = gamma_global[batch_index]
                local_gamma_variance[batch_index, feature_index] = variance_global[batch_index]
                local_delta_shape[batch_index, feature_index] = shape_global[batch_index]
                local_delta_scale[batch_index, feature_index] = scale_global[batch_index]
                batch_diagnostics.append({"fallback": "global_no_neighbors", "converged": False})
                continue
            weights /= weight_sum
            fitting_weights[batch_index, feature_index] = weights
            observation_variances = np.maximum(delta_hat[batch_index] / n_subjects, 1e-12)
            gamma_mean, gamma_variance, gamma_diagnostics = _fit_normal_prior(
                gamma_hat[batch_index],
                observation_variances,
                weights,
                float(gamma_global[batch_index]),
                float(variance_global[batch_index]),
                penalty,
            )
            delta_shape, delta_scale, delta_diagnostics = _fit_inverse_gamma_prior(
                delta_hat[batch_index],
                degrees,
                weights,
                float(shape_global[batch_index]),
                float(scale_global[batch_index]),
                penalty,
            )
            local_gamma_mean[batch_index, feature_index] = gamma_mean
            local_gamma_variance[batch_index, feature_index] = gamma_variance
            local_delta_shape[batch_index, feature_index] = delta_shape
            local_delta_scale[batch_index, feature_index] = delta_scale
            batch_diagnostics.append({"gamma": gamma_diagnostics, "delta": delta_diagnostics})
        diagnostics_by_target.append(batch_diagnostics)

    flat_diagnostics = [
        item for batch_items in diagnostics_by_target for item in batch_items
    ]
    gamma_diagnostics = [item.get("gamma", {}) for item in flat_diagnostics]
    delta_diagnostics = [item.get("delta", {}) for item in flat_diagnostics]
    optimizer_diagnostics = gamma_diagnostics + delta_diagnostics
    optimizer_fallbacks = sum(
        diagnostic.get("fallback") is not None
        for diagnostic in optimizer_diagnostics
    )
    optimizer_failures = sum(
        diagnostic.get("converged") is False
        for diagnostic in optimizer_diagnostics
    )

    return {
        "gamma_mean": local_gamma_mean,
        "gamma_variance": local_gamma_variance,
        "delta_shape": local_delta_shape,
        "delta_scale": local_delta_scale,
        "weights": weights_all,
        "fitting_weights": fitting_weights,
        "weight_diagnostics": weight_diagnostics,
        "target_diagnostics": diagnostics_by_target,
        "optimizer_fallback_count": int(optimizer_fallbacks),
        "optimizer_failure_count": int(optimizer_failures),
        "optimizer_count": int(len(optimizer_diagnostics)),
        "optimizer_fallback_rate": float(optimizer_fallbacks / max(1, len(optimizer_diagnostics))),
        "global_prior_penalty": penalty,
        "working_likelihood": "independent_feature_mean_errors_delta_over_n_and_gaussian_sample_variances",
    }
