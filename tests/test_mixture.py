from __future__ import annotations

import numpy as np

from simcombat import fit_combat
from simcombat.global_prior import construct_global_priors
from simcombat.mixture import fit_mixture_priors, initialize_memberships


def _two_group_similarity() -> tuple[np.ndarray, np.ndarray]:
    families = np.array(["left"] * 6 + ["right"] * 6)
    similarity = (families[:, None] == families[None, :]).astype(float)
    np.fill_diagonal(similarity, 0.0)
    return families, similarity


def test_structural_initialization_recovers_feature_families_softly() -> None:
    families, similarity = _two_group_similarity()
    memberships, diagnostics = initialize_memberships(
        12, 2, feature_families=families, custom_similarity=similarity, seed=12
    )
    assert np.allclose(memberships.sum(axis=1), 1.0)
    assert np.all(memberships >= 0)
    assert diagnostics["initialization"] == "structural_similarity_soft_kmeans++"
    left_mean = memberships[:6].mean(axis=0)
    right_mean = memberships[6:].mean(axis=0)
    assert max(np.dot(left_mean, right_mean), np.dot(left_mean, right_mean[::-1])) < 0.75


def test_k_one_matches_global_prior_parameters() -> None:
    rng = np.random.default_rng(311)
    gamma_hat = rng.normal(size=(3, 10))
    delta_hat = rng.lognormal(mean=0.0, sigma=0.25, size=(3, 10))
    batch_sizes = np.array([15, 18, 20])
    gamma_mean, gamma_variance, delta_shape, delta_scale, _ = construct_global_priors(
        gamma_hat, delta_hat
    )
    result = fit_mixture_priors(
        gamma_hat,
        delta_hat,
        batch_sizes,
        n_components=1,
        options={"max_iterations": 20},
    )
    assert result["memberships"].shape == (10, 1)
    assert np.allclose(result["memberships"], 1.0)
    assert np.allclose(result["gamma_mean"][:, 0], gamma_mean)
    assert np.allclose(result["gamma_variance"][:, 0], gamma_variance)
    assert np.allclose(result["delta_shape"][:, 0], delta_shape)
    assert np.allclose(result["delta_scale"][:, 0], delta_scale)
    assert result["diagnostics"]["converged"]


def test_mixture_fitter_normalizes_memberships_and_is_label_equivariant() -> None:
    rng = np.random.default_rng(12)
    n_features = 12
    families, similarity = _two_group_similarity()
    gamma_hat = np.vstack(
        [
            np.r_[rng.normal(-0.8, 0.15, 6), rng.normal(0.9, 0.15, 6)],
            np.r_[rng.normal(-0.6, 0.15, 6), rng.normal(0.7, 0.15, 6)],
            np.r_[rng.normal(-0.7, 0.15, 6), rng.normal(0.8, 0.15, 6)],
        ]
    )
    delta_hat = np.exp(
        np.vstack(
            [
                np.r_[rng.normal(-0.2, 0.08, 6), rng.normal(0.25, 0.08, 6)],
                np.r_[rng.normal(-0.1, 0.08, 6), rng.normal(0.2, 0.08, 6)],
                np.r_[rng.normal(-0.15, 0.08, 6), rng.normal(0.3, 0.08, 6)],
            ]
        )
    )
    options = {
        "seed": 12,
        "max_iterations": 40,
        "minimum_component_mass": 2.0,
        "global_prior_penalty": 0.2,
    }
    result = fit_mixture_priors(
        gamma_hat,
        delta_hat,
        np.array([24, 25, 27]),
        n_components=2,
        feature_families=families,
        similarity=similarity,
        options=options,
    )
    membership = result["memberships"]
    assert membership.shape == (n_features, 2)
    assert np.all(np.isfinite(membership))
    assert np.all(membership >= 0)
    assert np.allclose(membership.sum(axis=1), 1.0)
    assert np.all(np.isfinite(result["gamma_mean"]))
    assert np.all(np.isfinite(result["delta_shape"]))
    assert np.all(result["delta_shape"] > 2)
    assert result["diagnostics"]["component_mass"].min() >= 0

    reversed_result = fit_mixture_priors(
        gamma_hat,
        delta_hat,
        np.array([24, 25, 27]),
        n_components=2,
        memberships=membership[:, ::-1],
        feature_families=families,
        similarity=similarity,
        options=options,
    )
    assert np.allclose(reversed_result["memberships"], membership[:, ::-1], atol=0.15)


def test_public_mixture_mode_uses_quadrature_posterior() -> None:
    rng = np.random.default_rng(108)
    data = rng.normal(size=(4, 24))
    batch = np.repeat(["a", "b"], 12)
    data[0, 12:] += 0.4
    data[1, 12:] -= 0.25
    similarity = np.array(
        [
            [0.0, 1.0, 0.05, 0.05],
            [1.0, 0.0, 0.05, 0.05],
            [0.05, 0.05, 0.0, 1.0],
            [0.05, 0.05, 1.0, 0.0],
        ]
    )
    result = fit_combat(
        data,
        batch,
        prior_mode="mixture",
        similarity=similarity,
        options={"mixture_options": {"n_components": 2, "max_iterations": 20}},
    )
    assert result["harmonized"].shape == data.shape
    assert np.isfinite(result["harmonized"]).all()
    assert result["diagnostics"]["mixture_posterior"] is not None
    for batch_result in result["diagnostics"]["mixture_posterior"].values():
        responsibilities = batch_result["component_responsibilities"]
        assert np.all(np.isfinite(responsibilities))
        assert np.allclose(responsibilities.sum(axis=1), 1.0)
