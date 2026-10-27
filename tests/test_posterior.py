from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

from simcombat.posterior import iterative_posterior, mixture_posterior


def _single_component_reference(
    observations: np.ndarray,
    gamma_mean: float,
    gamma_variance: float,
    delta_shape: float,
    delta_scale: float,
) -> tuple[float, float, float]:
    n = observations.size
    sample_mean = float(np.mean(observations))
    centered_sum = float(np.sum((observations - sample_mean) ** 2))

    def unnormalized(delta: float, moment: str) -> float:
        conditional_variance = delta / n
        marginal_mean_variance = gamma_variance + conditional_variance
        conditional_gamma_variance = 1.0 / (1.0 / gamma_variance + n / delta)
        conditional_gamma_mean = conditional_gamma_variance * (
            gamma_mean / gamma_variance + n * sample_mean / delta
        )
        log_sample = -0.5 * n * np.log(2 * np.pi * delta) - 0.5 * centered_sum / delta
        log_mean = (
            -0.5 * np.log(2 * np.pi * marginal_mean_variance)
            -0.5 * (sample_mean - gamma_mean) ** 2 / marginal_mean_variance
        )
        log_prior = (
            delta_shape * np.log(delta_scale)
            - gammaln(delta_shape)
            - (delta_shape + 1) * np.log(delta)
            - delta_scale / delta
        )
        density = np.exp(log_sample + log_mean + log_prior)
        if moment == "delta":
            return delta * density
        if moment == "gamma":
            return conditional_gamma_mean * density
        return density

    evidence, _ = quad(lambda delta: unnormalized(delta, "evidence"), 0, np.inf, epsabs=1e-12, epsrel=1e-10)
    delta_numerator, _ = quad(lambda delta: unnormalized(delta, "delta"), 0, np.inf, epsabs=1e-12, epsrel=1e-10)
    gamma_numerator, _ = quad(lambda delta: unnormalized(delta, "gamma"), 0, np.inf, epsabs=1e-12, epsrel=1e-10)
    return evidence, gamma_numerator / evidence, delta_numerator / evidence


def test_component_evidence_and_posterior_means_match_independent_quadrature() -> None:
    observations = np.array([-0.7, 0.1, 0.4, 0.8])
    parameters = (0.15, 0.7, 4.2, 2.3)
    result = mixture_posterior(
        observations[None, :],
        np.ones((1, 1)),
        np.array([[parameters[0]]]),
        np.array([[parameters[1]]]),
        np.array([[parameters[2]]]),
        np.array([[parameters[3]]]),
        quadrature_bounds=(-25.0, 25.0),
    )
    evidence, gamma_mean, delta_mean = _single_component_reference(observations, *parameters)
    assert np.exp(result["component_log_evidence"][0, 0]) == pytest.approx(evidence, rel=1e-7)
    assert result["posterior_gamma_mean"][0] == pytest.approx(gamma_mean, rel=1e-7)
    assert result["posterior_delta_mean"][0] == pytest.approx(delta_mean, rel=1e-7)


def test_k_one_has_unit_responsibility_and_finite_posterior() -> None:
    observations = np.array([[0.1, 0.4, -0.2, 0.7], [-0.5, 0.2, 0.6, -0.1]])
    result = mixture_posterior(
        observations,
        np.ones((2, 1)),
        np.zeros((2, 1)),
        np.full((2, 1), 0.5),
        np.full((2, 1), 4.0),
        np.full((2, 1), 2.0),
    )
    assert np.allclose(result["component_responsibilities"], 1.0)
    assert np.isfinite(result["posterior_gamma_mean"]).all()
    assert np.isfinite(result["posterior_delta_mean"]).all()


def test_k_one_matches_conventional_iterative_posterior() -> None:
    observations = np.array(
        [[-0.3, 0.1, 0.7, 0.5, -0.6], [0.2, 0.4, -0.1, 0.8, 0.0]]
    )
    prior_mean = np.array([-0.1, 0.2])
    prior_variance = np.array([0.7, 0.5])
    prior_shape = np.array([4.5, 5.0])
    prior_scale = np.array([2.5, 3.0])
    mixture = mixture_posterior(
        observations,
        np.ones((2, 1)),
        prior_mean[:, None],
        prior_variance[:, None],
        prior_shape[:, None],
        prior_scale[:, None],
        quadrature_bounds=(-30.0, 30.0),
    )
    gamma_hat = observations.mean(axis=1)
    delta_hat = observations.var(axis=1, ddof=1)
    gamma_star, delta_star, diagnostics = iterative_posterior(
        observations,
        gamma_hat,
        delta_hat,
        prior_mean,
        prior_variance,
        prior_shape,
        prior_scale,
    )
    assert diagnostics["converged"]
    assert np.allclose(mixture["posterior_gamma_mean"], gamma_star, atol=0.08)
    assert np.allclose(mixture["posterior_delta_mean"], delta_star, atol=0.12)


def test_component_label_permutation_preserves_mixture_posterior() -> None:
    observations = np.array([[0.2, 0.1, 0.7, -0.3], [-0.1, 0.5, 0.3, 0.8]])
    memberships = np.array([[0.8, 0.2], [0.25, 0.75]])
    gamma_mean = np.array([[-0.2, 0.6], [0.1, -0.4]])
    gamma_variance = np.array([[0.3, 0.7], [0.5, 0.4]])
    delta_shape = np.array([[4.0, 6.0], [5.0, 3.5]])
    delta_scale = np.array([[3.0, 5.0], [4.0, 2.5]])
    first = mixture_posterior(
        observations, memberships, gamma_mean, gamma_variance, delta_shape, delta_scale
    )
    order = np.array([1, 0])
    second = mixture_posterior(
        observations,
        memberships[:, order],
        gamma_mean[:, order],
        gamma_variance[:, order],
        delta_shape[:, order],
        delta_scale[:, order],
    )
    assert np.allclose(first["component_responsibilities"], second["component_responsibilities"][:, order])
    assert np.allclose(first["posterior_gamma_mean"], second["posterior_gamma_mean"])
    assert np.allclose(first["posterior_delta_mean"], second["posterior_delta_mean"])
