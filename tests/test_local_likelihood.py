from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2, invgamma

from simcombat import fit_combat
from simcombat.local_likelihood import (
    _fit_inverse_gamma_prior,
    _fit_normal_prior,
    inverse_gamma_variance_log_likelihood,
    normal_mean_variance_log_likelihood,
)


def test_normal_marginal_likelihood_profiles_mean_and_is_finite() -> None:
    means = np.array([-0.4, 0.2, 0.7, 1.1])
    observation_variances = np.array([0.02, 0.08, 0.03, 0.1])
    weights = np.array([0.1, 0.3, 0.4, 0.2])
    log_likelihood, mean = normal_mean_variance_log_likelihood(
        means, observation_variances, weights, prior_mean=0.0, prior_variance=0.5
    )
    precision = weights / (observation_variances + 0.5)
    assert mean == pytest.approx(np.sum(precision * means) / precision.sum())
    assert np.isfinite(log_likelihood)


def test_normal_prior_optimizer_returns_finite_curvature() -> None:
    means = np.array([-0.2, 0.1, 0.0, 0.3, -0.1, 0.25])
    observation_variances = np.full(means.shape, 0.05)
    weights = np.full(means.shape, 1.0 / means.size)
    prior_mean, prior_variance, diagnostics = _fit_normal_prior(
        means, observation_variances, weights, 0.0, 0.4, penalty=0.1
    )
    assert np.isfinite(prior_mean)
    assert np.isfinite(prior_variance) and prior_variance > 0
    assert diagnostics["converged"]
    assert np.isfinite(diagnostics["curvature"])


def test_normal_optimizer_recovers_generating_parameters() -> None:
    rng = np.random.default_rng(818)
    observation_variances = np.full(2500, 0.03)
    true_mean = -0.25
    true_variance = 0.65
    observed_means = rng.normal(
        true_mean, np.sqrt(true_variance + observation_variances)
    )
    weights = np.full(observed_means.shape, 1.0 / observed_means.size)
    mean, variance, diagnostics = _fit_normal_prior(
        observed_means,
        observation_variances,
        weights,
        global_mean=0.0,
        global_variance=0.4,
        penalty=0.0,
    )
    assert diagnostics["converged"]
    assert mean == pytest.approx(true_mean, abs=0.08)
    assert variance == pytest.approx(true_variance, abs=0.1)


def test_normal_weighted_likelihood_is_scale_invariant_without_penalty() -> None:
    means = np.array([-0.4, -0.1, 0.2, 0.6, 0.9])
    observation_variances = np.full(means.shape, 0.04)
    weights = np.full(means.shape, 1.0 / means.size)
    first = _fit_normal_prior(means, observation_variances, weights, 0.0, 0.5, 0.0)
    second = _fit_normal_prior(means, observation_variances, weights * 7.0, 0.0, 0.5, 0.0)
    assert first[0] == pytest.approx(second[0], abs=1e-5)
    assert first[1] == pytest.approx(second[1], rel=1e-4)


def test_inverse_gamma_marginal_matches_independent_quadrature() -> None:
    variance = 0.8
    degrees = 6.0
    shape = 4.5
    scale = 2.2
    log_density = inverse_gamma_variance_log_likelihood(
        np.array([variance]), np.array([degrees]), np.array([1.0]), shape, scale
    )[0]
    integrated_density, _ = quad(
        lambda latent_variance: (
            chi2.pdf(degrees * variance / latent_variance, degrees)
            * degrees
            / latent_variance
            * invgamma.pdf(latent_variance, a=shape, scale=scale)
        ),
        0.0,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-10,
    )
    assert np.exp(log_density) == pytest.approx(integrated_density, rel=1e-9)


def test_inverse_gamma_optimizer_is_bounded_and_finite() -> None:
    variances = np.array([0.55, 0.7, 0.8, 1.0, 1.2, 1.35, 0.9, 1.1])
    degrees = np.full(variances.shape, 9.0)
    weights = np.full(variances.shape, 1.0 / variances.size)
    shape, scale, diagnostics = _fit_inverse_gamma_prior(
        variances, degrees, weights, global_shape=5.0, global_scale=3.0, penalty=0.1
    )
    assert shape > 2
    assert scale > 0
    assert np.isfinite(shape + scale)
    assert diagnostics["converged"]
    grid = inverse_gamma_variance_log_likelihood(
        variances, degrees, weights, shape, scale
    )
    assert np.isfinite(grid).all()


def test_inverse_gamma_optimizer_recovers_generating_parameters() -> None:
    rng = np.random.default_rng(919)
    true_shape, true_scale = 5.0, 4.0
    degrees = np.full(2500, 12.0)
    latent_variances = invgamma.rvs(
        a=true_shape, scale=true_scale, size=degrees.size, random_state=rng
    )
    sample_variances = latent_variances * chi2.rvs(
        degrees, size=degrees.size, random_state=rng
    ) / degrees
    weights = np.full(degrees.shape, 1.0 / degrees.size)
    shape, scale, diagnostics = _fit_inverse_gamma_prior(
        sample_variances,
        degrees,
        weights,
        global_shape=4.5,
        global_scale=3.5,
        penalty=0.0,
    )
    assert diagnostics["converged"]
    assert shape == pytest.approx(true_shape, abs=0.8)
    assert scale == pytest.approx(true_scale, abs=0.8)


def test_fit_combat_local_likelihood_smoke() -> None:
    rng = np.random.default_rng(421)
    batch = np.repeat(["scanner_a", "scanner_b"], 16)
    biological = rng.normal(size=(8, 32))
    observed = biological.copy()
    observed[:4, 16:] += np.array([[0.3], [-0.1], [0.2], [-0.25]])
    observed[4:, 16:] *= np.array([[1.1], [0.9], [1.2], [0.85]])
    result = fit_combat(
        observed,
        batch,
        prior_mode="local_likelihood",
        similarity=np.ones((8, 8)),
        options={
            "local_likelihood_options": {
                "global_prior_penalty": 0.2,
                "trust": 1.0,
            }
        },
    )
    assert result["harmonized"].shape == observed.shape
    assert result["diagnostics"]["prior_mode"] == "local_likelihood"
    assert all(
        item["converged"]
        for item in result["diagnostics"]["posterior"].values()
    )
    for key in ("harmonized", "gamma_star", "delta_star"):
        assert np.isfinite(result[key]).all()
    for key in ("gamma_mean", "gamma_variance", "delta_shape", "delta_scale"):
        assert np.isfinite(result["prior_details"][key]).all()
    fitting_weights = result["prior_details"]["fitting_weights"]
    assert np.allclose(np.diagonal(fitting_weights, axis1=1, axis2=2), 0.0)
    assert np.allclose(fitting_weights.sum(axis=2), 1.0)
    assert 0.0 <= result["prior_details"]["optimizer_fallback_rate"] <= 1.0
