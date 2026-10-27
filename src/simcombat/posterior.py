"""Conventional conditional empirical-Bayes posterior updates."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp


def iterative_posterior(
    standardized_batch: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    gamma_mean: np.ndarray,
    gamma_variance: np.ndarray,
    delta_shape: np.ndarray,
    delta_scale: np.ndarray,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Iterate conditional posterior means for one batch, feature-wise."""
    values = np.asarray(standardized_batch, dtype=float)
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    delta_hat = np.asarray(delta_hat, dtype=float)
    gamma_mean = np.asarray(gamma_mean, dtype=float)
    gamma_variance = np.asarray(gamma_variance, dtype=float)
    delta_shape = np.asarray(delta_shape, dtype=float)
    delta_scale = np.asarray(delta_scale, dtype=float)
    if values.ndim != 2 or values.shape[0] != gamma_hat.size:
        raise ValueError("standardized_batch must have shape (features, subjects).")
    if values.shape[1] < 2:
        raise ValueError("Each batch must contain at least two subjects.")
    if not (gamma_hat.shape == delta_hat.shape == gamma_mean.shape == gamma_variance.shape == delta_shape.shape == delta_scale.shape):
        raise ValueError("All posterior parameter arrays must have the same feature shape.")
    if np.any(gamma_variance <= 0) or np.any(delta_shape <= 1) or np.any(delta_scale <= 0):
        raise ValueError("Posterior priors require gamma_variance > 0, delta_shape > 1, and delta_scale > 0.")

    n_subjects = values.shape[1]
    gamma_old = gamma_hat.copy()
    delta_old = delta_hat.copy()
    converged = False
    change = np.inf
    for iteration in range(1, max_iterations + 1):
        gamma_new = (
            gamma_variance * n_subjects * gamma_hat + delta_old * gamma_mean
        ) / (gamma_variance * n_subjects + delta_old)
        residual_sum_squares = np.sum((values - gamma_new[:, None]) ** 2, axis=1)
        delta_new = (0.5 * residual_sum_squares + delta_scale) / (
            0.5 * n_subjects + delta_shape - 1.0
        )
        gamma_change = np.max(
            np.abs(gamma_new - gamma_old) / np.maximum(1.0, np.abs(gamma_old))
        )
        delta_change = np.max(
            np.abs(delta_new - delta_old) / np.maximum(1.0, np.abs(delta_old))
        )
        change = float(max(gamma_change, delta_change))
        gamma_old, delta_old = gamma_new, delta_new
        if change <= tolerance:
            converged = True
            break

    if not np.all(np.isfinite(gamma_old)) or not np.all(np.isfinite(delta_old)):
        raise FloatingPointError("Posterior iteration produced non-finite estimates.")
    return gamma_old, delta_old, {
        "iterations": iteration,
        "converged": converged,
        "final_relative_change": change,
    }


def mixture_posterior(
    standardized_batch: np.ndarray,
    memberships: np.ndarray,
    gamma_mean: np.ndarray,
    gamma_variance: np.ndarray,
    delta_shape: np.ndarray,
    delta_scale: np.ndarray,
    *,
    quadrature_bounds: tuple[float, float] = (-30.0, 30.0),
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-8,
) -> dict[str, np.ndarray | dict[str, object]]:
    """Integrate the joint Normal/inverse-gamma posterior over each component.

    For each component, gamma is analytically integrated conditional on delta;
    adaptive quadrature then integrates over log(delta), including its Jacobian.
    The returned correction uses component-averaged posterior means. This is a
    plug-in ComBat transformation, not the full posterior predictive transform.
    """
    values = np.asarray(standardized_batch, dtype=float)
    memberships = np.asarray(memberships, dtype=float)
    gamma_mean = np.asarray(gamma_mean, dtype=float)
    gamma_variance = np.asarray(gamma_variance, dtype=float)
    delta_shape = np.asarray(delta_shape, dtype=float)
    delta_scale = np.asarray(delta_scale, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("standardized_batch must have shape (features, subjects), n >= 2.")
    n_features, n_subjects = values.shape
    if memberships.ndim != 2 or memberships.shape[0] != n_features:
        raise ValueError("memberships must have shape (features, components).")
    n_components = memberships.shape[1]
    expected_shape = (n_features, n_components)
    if any(array.shape != expected_shape for array in (gamma_mean, gamma_variance, delta_shape, delta_scale)):
        raise ValueError("Mixture prior arrays must share shape (features, components).")
    if not all(np.all(np.isfinite(array)) for array in (values, memberships, gamma_mean, gamma_variance, delta_shape, delta_scale)):
        raise ValueError("Mixture posterior inputs must be finite.")
    if np.any(memberships < 0) or not np.allclose(memberships.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Each feature membership row must be nonnegative and sum to one.")
    if np.any(gamma_variance <= 0) or np.any(delta_shape <= 0) or np.any(delta_scale <= 0):
        raise ValueError("Mixture priors require positive variances, shapes, and scales.")
    lower, upper = quadrature_bounds
    if not np.isfinite(lower + upper) or lower >= upper:
        raise ValueError("quadrature_bounds must be finite and increasing.")

    sample_mean = np.mean(values, axis=1)
    centered_sum_squares = np.sum((values - sample_mean[:, None]) ** 2, axis=1)
    component_log_evidence = np.full(expected_shape, -np.inf)
    component_gamma_mean = np.zeros(expected_shape)
    component_delta_mean = np.zeros(expected_shape)
    component_gamma_variance = np.zeros(expected_shape)
    integration_errors = np.zeros(expected_shape)

    for feature_index in range(n_features):
        sample_variance_term = centered_sum_squares[feature_index]
        xbar = sample_mean[feature_index]
        for component_index in range(n_components):
            prior_mean = gamma_mean[feature_index, component_index]
            prior_variance = gamma_variance[feature_index, component_index]
            shape = delta_shape[feature_index, component_index]
            scale = delta_scale[feature_index, component_index]

            def log_joint(log_delta: float) -> float:
                delta = float(np.exp(log_delta))
                conditional_mean_variance = delta / n_subjects
                marginal_mean_variance = prior_variance + conditional_mean_variance
                gamma_conditional_variance = 1.0 / (
                    1.0 / prior_variance + n_subjects / delta
                )
                gamma_conditional_mean = gamma_conditional_variance * (
                    prior_mean / prior_variance + n_subjects * xbar / delta
                )
                log_sample_likelihood = (
                    -0.5 * n_subjects * np.log(2.0 * np.pi * delta)
                    -0.5 * sample_variance_term / delta
                )
                log_mean_marginal = (
                    -0.5 * np.log(2.0 * np.pi * marginal_mean_variance)
                    -0.5 * (xbar - prior_mean) ** 2 / marginal_mean_variance
                )
                log_inverse_gamma = (
                    shape * np.log(scale)
                    - gammaln(shape)
                    - (shape + 1.0) * log_delta
                    - scale / delta
                )
                return float(
                    log_sample_likelihood
                    + log_mean_marginal
                    + log_inverse_gamma
                    + log_delta
                )

            from scipy.special import gammaln

            points = np.linspace(lower, upper, 1201)
            log_values = np.array([log_joint(float(point)) for point in points])
            log_scale_value = float(np.max(log_values))
            if not np.isfinite(log_scale_value):
                continue

            def scaled_density(log_delta: float) -> float:
                return float(np.exp(log_joint(log_delta) - log_scale_value))

            denominator, error = quad(
                scaled_density,
                lower,
                upper,
                epsabs=absolute_tolerance,
                epsrel=relative_tolerance,
                limit=300,
            )
            if denominator <= 0 or not np.isfinite(denominator):
                continue
            delta_numerator, delta_error = quad(
                lambda log_delta: np.exp(log_delta) * scaled_density(log_delta),
                lower,
                upper,
                epsabs=absolute_tolerance,
                epsrel=relative_tolerance,
                limit=300,
            )
            gamma_numerator, gamma_error = quad(
                lambda log_delta: (
                    1.0
                    / (1.0 / prior_variance + n_subjects / np.exp(log_delta))
                    * (
                        prior_mean / prior_variance
                        + n_subjects * xbar / np.exp(log_delta)
                    )
                    * scaled_density(log_delta)
                ),
                lower,
                upper,
                epsabs=absolute_tolerance,
                epsrel=relative_tolerance,
                limit=300,
            )
            gamma_second_numerator, second_error = quad(
                lambda log_delta: (
                    (
                        1.0
                        / (1.0 / prior_variance + n_subjects / np.exp(log_delta))
                        * (
                            prior_mean / prior_variance
                            + n_subjects * xbar / np.exp(log_delta)
                        )
                    ) ** 2
                    + 1.0 / (1.0 / prior_variance + n_subjects / np.exp(log_delta))
                )
                * scaled_density(log_delta),
                lower,
                upper,
                epsabs=absolute_tolerance,
                epsrel=relative_tolerance,
                limit=300,
            )
            component_log_evidence[feature_index, component_index] = log_scale_value + np.log(denominator)
            component_delta_mean[feature_index, component_index] = delta_numerator / denominator
            component_gamma_mean[feature_index, component_index] = gamma_numerator / denominator
            gamma_second = gamma_second_numerator / denominator
            component_gamma_variance[feature_index, component_index] = max(
                0.0, gamma_second - component_gamma_mean[feature_index, component_index] ** 2
            )
            integration_errors[feature_index, component_index] = (
                error + delta_error + gamma_error + second_error
            )

    log_joint_component = np.log(np.maximum(memberships, np.finfo(float).tiny)) + component_log_evidence
    log_normalizer = logsumexp(log_joint_component, axis=1, keepdims=True)
    responsibilities = np.exp(log_joint_component - log_normalizer)
    invalid = ~np.isfinite(log_normalizer[:, 0])
    if np.any(invalid):
        responsibilities[invalid] = memberships[invalid]
        responsibilities[invalid] /= responsibilities[invalid].sum(axis=1, keepdims=True)

    posterior_gamma_mean = np.sum(responsibilities * component_gamma_mean, axis=1)
    posterior_delta_mean = np.sum(responsibilities * component_delta_mean, axis=1)
    posterior_gamma_variance = np.sum(
        responsibilities
        * (
            component_gamma_variance
            + (component_gamma_mean - posterior_gamma_mean[:, None]) ** 2
        ),
        axis=1,
    )
    return {
        "component_log_evidence": component_log_evidence,
        "component_responsibilities": responsibilities,
        "component_gamma_mean": component_gamma_mean,
        "component_delta_mean": component_delta_mean,
        "posterior_gamma_mean": posterior_gamma_mean,
        "posterior_delta_mean": posterior_delta_mean,
        "posterior_gamma_variance": posterior_gamma_variance,
        "integration_error": integration_errors,
        "diagnostics": {
            "quadrature_bounds": quadrature_bounds,
            "invalid_evidence_rows": invalid,
            "correction_type": "posterior_mean_plugin_not_posterior_predictive",
        },
    }
