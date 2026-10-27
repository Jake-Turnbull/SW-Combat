"""Soft-membership mixture priors over batch effects."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .global_prior import construct_global_priors
from .similarity import build_similarity


def initialize_memberships(
    n_features: int,
    n_components: int,
    *,
    feature_families: object | None = None,
    overlap: np.ndarray | None = None,
    custom_similarity: np.ndarray | None = None,
    temperature: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create deterministic soft clusters from structural, non-effect features."""
    if n_components < 1 or n_components > n_features:
        raise ValueError("n_components must lie between one and n_features.")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite.")
    if n_components == 1:
        return np.ones((n_features, 1)), {"initialization": "single_component"}
    similarity, diagnostics = build_similarity(
        n_features,
        feature_families=feature_families,
        overlap=overlap,
        custom_similarity=custom_similarity,
        trust=1.0,
        exclude_diagonal=True,
    )
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(n_features))]
    for _ in range(1, n_components):
        distances = np.maximum(1.0 - similarity[:, seeds].max(axis=1), 0.0)
        distances[seeds] = 0.0
        probabilities = distances**2
        if probabilities.sum() == 0:
            candidate = next(index for index in range(n_features) if index not in seeds)
        else:
            candidate = int(rng.choice(n_features, p=probabilities / probabilities.sum()))
        seeds.append(candidate)
    distances = 1.0 - similarity[:, seeds]
    logits = -distances / temperature
    logits -= logsumexp(logits, axis=1, keepdims=True)
    memberships = np.exp(logits)
    return memberships, {
        "initialization": "structural_similarity_soft_kmeans++",
        "seed_features": np.asarray(seeds),
        "temperature": float(temperature),
        **diagnostics,
    }


def _update_normal_component(
    observations: np.ndarray,
    observation_variances: np.ndarray,
    responsibilities: np.ndarray,
    previous_mean: float,
    previous_variance: float,
    penalty: float,
) -> tuple[float, float, bool]:
    active = responsibilities > 1e-10
    if active.sum() < 2:
        return previous_mean, previous_variance, False
    values = observations[active]
    measurement_variance = observation_variances[active]
    weights = responsibilities[active]
    lower = np.log(max(float(np.min(measurement_variance)) * 1e-6, 1e-8))
    upper = np.log(max(float(np.var(values)), previous_variance, 1.0) * 1e4)

    def objective(parameters: np.ndarray) -> float:
        mean = float(parameters[0])
        variance = float(np.exp(parameters[1]))
        total_variance = variance + measurement_variance
        log_likelihood = -0.5 * np.sum(
            weights
            * (
                np.log(total_variance)
                + (values - mean) ** 2 / total_variance
            )
        )
        penalty_value = 0.5 * penalty * (
            (mean - previous_mean) ** 2 / max(previous_variance, 1e-8)
            + (parameters[1] - np.log(max(previous_variance, 1e-8))) ** 2
        )
        return float(-log_likelihood + penalty_value)

    result = minimize(
        objective,
        np.array([previous_mean, np.log(max(previous_variance, 1e-8))]),
        method="L-BFGS-B",
        bounds=[(None, None), (lower, upper)],
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return previous_mean, previous_variance, False
    return float(result.x[0]), float(np.exp(result.x[1])), True


def fit_mixture_priors(
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    batch_sizes: np.ndarray,
    n_components: int,
    *,
    memberships: np.ndarray | None = None,
    feature_families: object | None = None,
    overlap: np.ndarray | None = None,
    similarity: np.ndarray | None = None,
    options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit a shared-membership mixture with batch-specific Normal/IG priors.

    This generalized-EM working fit uses raw gamma and delta estimates with their
    sampling variances. It does not treat true group labels as inputs. Mixture
    labels couple additive and multiplicative prior components by design.
    """
    options = {} if options is None else dict(options)
    gamma_hat = np.asarray(gamma_hat, dtype=float)
    delta_hat = np.asarray(delta_hat, dtype=float)
    batch_sizes = np.asarray(batch_sizes, dtype=float)
    if gamma_hat.ndim != 2 or delta_hat.shape != gamma_hat.shape:
        raise ValueError("gamma_hat and delta_hat must share shape (batches, features).")
    n_batches, n_features = gamma_hat.shape
    if batch_sizes.shape != (n_batches,) or np.any(batch_sizes < 2):
        raise ValueError("batch_sizes must contain at least two subjects for each batch.")
    if not np.all(np.isfinite(gamma_hat)) or not np.all(np.isfinite(delta_hat)) or np.any(delta_hat <= 0):
        raise ValueError("Batch effect estimates must be finite and delta_hat strictly positive.")
    if n_components < 1 or n_components > n_features:
        raise ValueError("n_components must lie between one and n_features.")
    if memberships is None:
        memberships, initialization_diagnostics = initialize_memberships(
            n_features,
            n_components,
            feature_families=feature_families,
            overlap=overlap,
            custom_similarity=similarity,
            temperature=float(options.get("initialization_temperature", 0.5)),
            seed=int(options.get("seed", 0)),
        )
    else:
        memberships = np.asarray(memberships, dtype=float)
        initialization_diagnostics = {"initialization": "user_supplied_memberships"}
        if memberships.shape != (n_features, n_components):
            raise ValueError("memberships must have shape (features, components).")
        if np.any(memberships < 0) or not np.all(np.isfinite(memberships)):
            raise ValueError("memberships must be finite and nonnegative.")
        row_totals = memberships.sum(axis=1, keepdims=True)
        if np.any(row_totals <= 0):
            raise ValueError("Each feature must have positive total membership.")
        memberships = memberships / row_totals

    global_gamma_mean, global_gamma_variance, global_delta_shape, global_delta_scale, _ = (
        construct_global_priors(gamma_hat, delta_hat)
    )
    if n_components == 1:
        return {
            "memberships": np.ones((n_features, 1)),
            "gamma_mean": global_gamma_mean[:, None],
            "gamma_variance": global_gamma_variance[:, None],
            "delta_shape": global_delta_shape[:, None],
            "delta_scale": global_delta_scale[:, None],
            "diagnostics": {
                **initialization_diagnostics,
                "iterations": 0,
                "converged": True,
                "membership_change_history": np.array([]),
                "observed_log_likelihood_history": np.array([]),
                "component_mass": np.array([float(n_features)]),
                "collapsed_components": np.array([False]),
                "successful_parameter_updates": 0,
                "membership_model": "shared_feature_membership_for_location_and_scale",
                "graph_regularization": "not_used",
                "equivalent_to_global_component_prior": True,
            },
        }
    gamma_mean = np.repeat(global_gamma_mean[:, None], n_components, axis=1)
    gamma_variance = np.repeat(global_gamma_variance[:, None], n_components, axis=1)
    delta_shape = np.repeat(global_delta_shape[:, None], n_components, axis=1)
    delta_scale = np.repeat(global_delta_scale[:, None], n_components, axis=1)
    max_iterations = int(options.get("max_iterations", 100))
    tolerance = float(options.get("tolerance", 1e-5))
    minimum_mass = float(options.get("minimum_component_mass", max(1.0, 0.02 * n_features)))
    penalty = float(options.get("global_prior_penalty", 0.1))
    if max_iterations < 1 or tolerance <= 0 or minimum_mass <= 0 or penalty < 0:
        raise ValueError("Invalid mixture optimization controls.")

    mean_observation_variance = delta_hat / batch_sizes[:, None]
    degrees = batch_sizes - 1.0
    objective_history: list[float] = []
    convergence_history: list[float] = []
    collapsed = np.zeros(n_components, dtype=bool)

    for iteration in range(1, max_iterations + 1):
        old_memberships = memberships.copy()
        log_responsibilities = np.zeros((n_features, n_components))
        for feature_index in range(n_features):
            for component_index in range(n_components):
                log_likelihood = 0.0
                for batch_index in range(n_batches):
                    mean_variance = (
                        gamma_variance[batch_index, component_index]
                        + mean_observation_variance[batch_index, feature_index]
                    )
                    log_likelihood += -0.5 * (
                        np.log(mean_variance)
                        + (
                            gamma_hat[batch_index, feature_index]
                            - gamma_mean[batch_index, component_index]
                        )
                        ** 2
                        / mean_variance
                    )
                    shape = delta_shape[batch_index, component_index]
                    scale = delta_scale[batch_index, component_index]
                    df_half = degrees[batch_index] / 2.0
                    variance = delta_hat[batch_index, feature_index]
                    log_likelihood += (
                        gammaln(shape + df_half)
                        - gammaln(shape)
                        - gammaln(df_half)
                        + df_half * np.log(df_half)
                        + shape * np.log(scale)
                        + (df_half - 1.0) * np.log(variance)
                        - (shape + df_half) * np.log(scale + df_half * variance)
                    )
                log_responsibilities[feature_index, component_index] = (
                    np.log(max(memberships[feature_index, component_index], np.finfo(float).tiny))
                    + log_likelihood
                )
        log_normalizer = logsumexp(log_responsibilities, axis=1, keepdims=True)
        memberships = np.exp(log_responsibilities - log_normalizer)
        masses = memberships.sum(axis=0)
        collapsed = masses < minimum_mass
        if np.any(collapsed):
            memberships[:, collapsed] = np.maximum(memberships[:, collapsed], 1e-8)
            memberships /= memberships.sum(axis=1, keepdims=True)
            masses = memberships.sum(axis=0)

        successful_updates = 0
        for batch_index in range(n_batches):
            for component_index in range(n_components):
                responsibility = memberships[:, component_index]
                updated_mean, updated_variance, converged = _update_normal_component(
                    gamma_hat[batch_index],
                    mean_observation_variance[batch_index],
                    responsibility,
                    float(global_gamma_mean[batch_index]),
                    float(global_gamma_variance[batch_index]),
                    penalty,
                )
                gamma_mean[batch_index, component_index] = updated_mean
                gamma_variance[batch_index, component_index] = updated_variance
                successful_updates += int(converged)

                active = responsibility > 1e-10
                if np.sum(active) >= 2 and masses[component_index] >= minimum_mass:
                    local_weights = responsibility[active]
                    local_weights /= local_weights.sum()
                    local_shape, local_scale, diagnostics = _fit_inverse_gamma_component(
                        delta_hat[batch_index, active],
                        np.full(np.sum(active), degrees[batch_index]),
                        local_weights,
                        float(global_delta_shape[batch_index]),
                        float(global_delta_scale[batch_index]),
                        penalty,
                    )
                    delta_shape[batch_index, component_index] = local_shape
                    delta_scale[batch_index, component_index] = local_scale
                    successful_updates += int(bool(diagnostics["converged"]))
                else:
                    delta_shape[batch_index, component_index] = global_delta_shape[batch_index]
                    delta_scale[batch_index, component_index] = global_delta_scale[batch_index]

        change = float(np.max(np.abs(memberships - old_memberships)))
        convergence_history.append(change)
        objective_history.append(float(np.sum(log_normalizer)))
        if change <= tolerance:
            break

    diagnostics = {
        **initialization_diagnostics,
        "iterations": iteration,
        "converged": bool(convergence_history[-1] <= tolerance),
        "membership_change_history": np.asarray(convergence_history),
        "observed_log_likelihood_history": np.asarray(objective_history),
        "component_mass": memberships.sum(axis=0),
        "collapsed_components": collapsed,
        "successful_parameter_updates": successful_updates,
        "membership_model": "shared_feature_membership_for_location_and_scale",
        "graph_regularization": "not_used",
    }
    return {
        "memberships": memberships,
        "gamma_mean": gamma_mean,
        "gamma_variance": gamma_variance,
        "delta_shape": delta_shape,
        "delta_scale": delta_scale,
        "diagnostics": diagnostics,
    }


def _fit_inverse_gamma_component(
    sample_variances: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: np.ndarray,
    prior_shape: float,
    prior_scale: float,
    penalty: float,
) -> tuple[float, float, dict[str, object]]:
    from .local_likelihood import _fit_inverse_gamma_prior

    return _fit_inverse_gamma_prior(
        sample_variances,
        degrees_of_freedom,
        weights,
        prior_shape,
        prior_scale,
        penalty,
    )
